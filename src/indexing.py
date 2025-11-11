import os
import json
import faiss
import nltk
import ssl
import string
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List
import re 

import config

# --- CẤU HÌNH MODEL EMBEDDING ---
try:
    print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}...")
    GLOBAL_MODEL = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}.")
    GLOBAL_MODEL = None

# --- CẤU HÌNH NLTK ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
english_stopwords = set(stopwords.words('english'))

## ================ MAIN CODE ==================== ##

def generate_single_embedding(text: str) -> np.ndarray:
    """
    Chuyển đổi một chuỗi văn bản (tóm tắt CV) thành vector embedding 768 chiều.
    """
    if GLOBAL_MODEL is None or not text or not isinstance(text, str):
        print("Model not loaded or input is invalid.")
        return np.array([])

    # Model.encode luôn nhận input là một list, ngay cả khi chỉ có 1 phần tử
    try:
        embeddings = GLOBAL_MODEL.encode([text], convert_to_tensor=False, show_progress_bar=False)
        return embeddings[0] # Trả về vector 1D
    except Exception as e:
        print(f"Error during encoding: {e}")
        return np.array([])
    
def process_summaries_to_embeddings(summary_folder: str) -> dict:
    """
    Đọc nội dung từng file .summary, tạo vector embedding và lưu vào dictionary.
    """
    if GLOBAL_MODEL is None:
        print("Cannot run, embedding model is not loaded.")
        return {}

    all_embeddings = {}

    # Lấy danh sách tất cả các file trong thư mục và chỉ lọc lấy file .summary
    summary_files = [f for f in os.listdir(summary_folder) if f.endswith('.summary')]

    print(f"\nFound {len(summary_files)} summary files in {summary_folder}. Starting embedding...")

    for filename in tqdm(summary_files):
        filepath = os.path.join(summary_folder, filename)

        try:
            # 1. Đọc nội dung file summary
            with open(filepath, 'r', encoding='utf-8') as f:
                summary_text = f.read()

            # 2. Tạo vector embedding cho nội dung tóm tắt
            embedding_vector = generate_single_embedding(summary_text)

            if embedding_vector.size > 0:
                # 3. Lưu vector vào dictionary với tên file làm key
                all_embeddings[filename] = embedding_vector

                # OPTIONAL: Lưu vector ra file .npy riêng biệt nếu cần
                # np.save(os.path.join(OUTPUT_EMBEDDINGS_FOLDER, filename.replace('.summary', '.npy')), embedding_vector)

            else:
                print(f"Skipped {filename}: Could not generate vector.")

        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            all_embeddings[filename] = None # Đánh dấu lỗi

    print("\n✅ Embedding process complete.")
    return all_embeddings

def build_rich_faiss_index_offline(summary_folder: str, json_folder: str, pdf_folder: str, output_folder: str, dimension: int):
    """
    Xây dựng FAISS Index và lưu trữ ánh xạ giàu thông tin (Summary, PDF File, Index ID, và dữ liệu JSON).
    """
    if GLOBAL_MODEL is None:
        print("FATAL: Embedding model not loaded. Cannot build index.")
        return False

    all_vectors = []
    rich_metadata = []

    summary_files = [f for f in os.listdir(summary_folder) if f.endswith('.summary')]
    print(f"\nFound {len(summary_files)} summary files. Starting rich index building...")

    for filename in tqdm(summary_files, desc="Embedding and Preparing Metadata"):
        base_name = os.path.splitext(filename)[0]
        summary_filepath = os.path.join(summary_folder, filename)
        json_filepath = os.path.join(json_folder, f"{base_name}.json")
        pdf_filepath = os.path.join(pdf_folder, f"{base_name}.pdf")

        try:
            # 1. Đọc nội dung Summary
            with open(summary_filepath, 'r', encoding='utf-8') as f:
                summary_text = f.read()

            # 2. Tạo vector
            vector = generate_single_embedding(summary_text)

            # 3. KIỂM TRA TÍNH HỢP LỆ CỦA VECTOR (kích thước phải là 768)
            if vector.size != dimension:
                if vector.size > 0:
                    print(f"Warning: Vector for {filename} has size {vector.size}, expected {dimension}. Skipped.")
                else:
                    print(f"Warning: Failed to generate vector for {filename}. Skipped.")
                continue

            # 4. Đọc dữ liệu JSON
            try:
                with open(json_filepath, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
            except Exception as e:
                print(f"Warning: Could not read JSON {json_filepath}. Error: {e}. Skipping.")
                continue

            # 5. Thêm vector vào danh sách
            all_vectors.append(vector)

            # 6. Xây dựng Metadata (Gộp dữ liệu)
            metadata_item = {
                "summary_filename": filename,
                "pdf_filepath": pdf_filepath,
                "summary_content": summary_text,
                "skills": json_data.get("skills", []),
                "industry": json_data.get("detected_industry", "unknown"),
                "job_titles": json_data.get("normalized_job_titles", []),
                "yoe": json_data.get("years_of_experience", 0)
            }
            rich_metadata.append(metadata_item)

        except Exception as e:
            print(f"Warning: Failed to process {filename}. Error: {e}")

    if not all_vectors:
        print("No valid vectors were generated. Indexing aborted.")
        return False

    vectors_array = np.array(all_vectors).astype('float32')

    if len(vectors_array.shape) != 2 or vectors_array.shape[1] != dimension:
        print(f"FATAL: Final vectors_array shape is incorrect: {vectors_array.shape}. Expected (N, {dimension}). Indexing aborted.")
        return False

    print(f"Batch embedding complete. Starting FAISS Indexing with {vectors_array.shape[0]} vectors...")

    # Xử lý Index FAISS
    faiss.normalize_L2(vectors_array)
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors_array)

    # 3. Lưu Index và Ánh xạ mới
    index_path = os.path.join(output_folder, config.FAISS_INDEX_FILE)
    map_path = os.path.join(output_folder, config.ID_MAP_FILE)

    faiss.write_index(index, index_path)

    with open(map_path, 'w', encoding='utf-8') as f:
        json.dump(rich_metadata, f, indent=4)

    print(f"\n✅ FAISS Index saved to: {index_path}")
    print(f"✅ Rich ID Map (Skills/Industry/Summary/PDF) saved to: {map_path}")
    return True

def preprocess_text(text):
    """Tiền xử lý văn bản: chữ thường, loại bỏ dấu câu, stopwords, và lemmatization."""
    # 1. Chuyển chữ thường và loại bỏ dấu câu
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 2. Tokenization và Lemmatization
    tokens = text.split()
    processed_tokens = []
    for token in tokens:
        # Loại bỏ Stopwords
        if token not in english_stopwords:
            # Lemmatization (Đưa từ về dạng gốc)
            processed_tokens.append(lemmatizer.lemmatize(token))

    return processed_tokens

def tokenize_with_skills(text: str, skills: List[str] = None):
    tokens = preprocess_text(text)
    if skills:
        # add multiword skill tokens (no punctuation, joined by underscore)
        for s in skills:
            s_norm = re.sub(r'[^a-z0-9\s]', '', s.lower()).strip()
            if len(s_norm.split()) > 1:
                tokens.append("_".join(s_norm.split()))
    return tokens

def build_bm25_index(map_path: str, bm25_index_path: str):
    """
    Tải Rich Metadata Map và xây dựng BM25 Index.
    Đã chỉnh sửa để nhận đường dẫn làm tham số.
    """
    print(f"Bắt đầu xây dựng BM25 Index từ file: {map_path}")
    try:
        with open(map_path, 'r', encoding='utf-8') as f:
            rich_metadata = json.load(f)
    except Exception as e:
        print(f"Lỗi khi tải Rich Metadata Map: {e}")
        return None, None

    tokenized_corpus = []
    
    for i, item in enumerate(rich_metadata):
        item['doc_id'] = i
        raw_text = item.get("summary_content", "")
        skills = item.get("skills", [])
        
        processed_tokens = tokenize_with_skills(raw_text, skills=skills)
        tokenized_corpus.append(processed_tokens)

    if not tokenized_corpus:
        print("Lỗi: Corpus rỗng.")
        return None, None

    bm25 = BM25Okapi(tokenized_corpus)
    print(f"Đã xây dựng BM25 Index thành công cho {len(tokenized_corpus)} tài liệu.")

    storage_data = {
        "tokenized_corpus": tokenized_corpus,
        "rich_metadata_with_id": rich_metadata
    }

    with open(bm25_index_path, 'w', encoding='utf-8') as f:
        json.dump(storage_data, f, indent=2)

    print(f"✅ BM25 Corpus đã được lưu trữ tại: {bm25_index_path}")
    return bm25, rich_metadata