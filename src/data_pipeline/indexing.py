import os, re
import json
import faiss
import nltk
import ssl
import numpy as np

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List

import config

from src.core.ontology import skill_ontology

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

    try:
        embeddings = GLOBAL_MODEL.encode([text], convert_to_tensor=False, show_progress_bar=False)
        return embeddings[0] 
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

    summary_files = [f for f in os.listdir(summary_folder) if f.endswith('.summary')]

    print(f"\nFound {len(summary_files)} summary files in {summary_folder}. Starting embedding...")

    for filename in tqdm(summary_files):
        filepath = os.path.join(summary_folder, filename)

        try:
            # 1. Đọc nội dung file summary
            with open(filepath, 'r', encoding='utf-8') as f:
                summary_text = f.read()

            embedding_vector = generate_single_embedding(summary_text)

            if embedding_vector.size > 0:
                all_embeddings[filename] = embedding_vector

            else:
                print(f"Skipped {filename}: Could not generate vector.")

        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            all_embeddings[filename] = None

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
    
    for filename in tqdm(summary_files, desc="Indexing"):
        base_name = os.path.splitext(filename)[0]
        summary_path = os.path.join(summary_folder, filename)
        json_path = os.path.join(json_folder, f"{base_name}.json")
        pdf_path = os.path.join(pdf_folder, f"{base_name}.pdf")
        
        try:
            # 1. Embed Summary
            with open(summary_path, 'r', encoding='utf-8') as f:
                text = f.read()
            vector = generate_single_embedding(text)
            
            # 2. Read JSON Metadata
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 3. Extract New Fields
            profile = data.get("candidate_profile", {})
            metrics = data.get("metrics", {})
            flat_skills = flatten_tech_stack(data)
            
            meta = {
                "id": base_name,
                "summary_filename": filename,
                "pdf_filepath": f"{base_name}.pdf",
                "summary_content": text,
                "name": profile.get("name", "N/A"),
                "role": profile.get("role_focus", "N/A"),
                "seniority": profile.get("seniority_level", "N/A"),
                "yoe": metrics.get("years_experience", 0),
                "english_level": metrics.get("english_cefr_level", "N/A"),
                "certifications": data.get("certifications", []),
                "awards": data.get("honors_and_awards", []),
                "skills": flat_skills, 
                "tech_stack": data.get("tech_stack", {}) 
            }
            
            rich_metadata.append(meta)
            all_vectors.append(vector)
            
        except Exception as e:
            print(f"Skip {filename}: {e}")

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
    if not text or not isinstance(text, str): return []
    
    text = text.lower()
    text = re.sub(r'http\S+', '', text) # Bỏ link
    
    # --- DÒNG QUAN TRỌNG NHẤT ---
    # Chỉ giữ lại a-z và 0-9. Mọi ký tự lạ như \uff8a, tiếng Việt có dấu, %^&... sẽ biến mất
    text = re.sub(r'[^a-z0-9]', ' ', text) 
    # ----------------------------

    tokens = text.split()
    processed_tokens = []
    
    # Danh sách đen (Chặn đứng 'frid', 'dto')
    BLACKLIST = {"frid", "dto", "consequ", "actiontypes", "uct", "dee", "ieren", "cid"}

    for token in tokens:
        if len(token) < 2 or len(token) > 20: continue
        if token in english_stopwords or token in BLACKLIST: continue
        
        # Chặn các từ dính chùm vô nghĩa
        if "ieren" in token or "uct" in token: continue 

        lemma = lemmatizer.lemmatize(token)
        if len(lemma) > 1:
            processed_tokens.append(lemma)

    return processed_tokens

def flatten_tech_stack(json_data):
    """
    Gom nhóm skill và CHUẨN HÓA tên gọi dựa trên Ontology.
    """
    tech_stack = json_data.get("tech_stack", {})
    all_skills = []
    
    # Duyệt qua các category
    for category in tech_stack.values():
        if isinstance(category, list):
            for skill in category:
                # --- Chuẩn hóa skill ---
                normalized_name = skill_ontology.normalize_skill(skill)
                all_skills.append(normalized_name)
                # ----------------------------------
            
    languages = json_data.get("languages", [])
    for lang in languages:
        certs = lang.get("certifications_and_equivalents")
        if certs and isinstance(certs, list):
            all_skills.extend(certs)

    return list(set(all_skills))

def tokenize_with_skills(text: str, skills: list = None):
    """
    Tokenize văn bản và thêm các skill (n-grams) vào corpus.
    """
    # 1. Xử lý văn bản chính
    tokens = preprocess_text(text)
    
    # 2. Thêm Skills 
    if skills:
        for s in skills:
            if not s or not isinstance(s, str): continue      
            s_clean = s.lower().strip()
            
            if ' ' in s_clean:
                s_joined = re.sub(r'\s+', '_', s_clean)
                tokens.append(s_joined)
            
            s_norm = re.sub(r'[^a-z0-9]', '', s_clean)
            if len(s_norm) > 1:
                tokens.append(s_norm)

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