import os
import re
import ssl
import json
import faiss
import nltk
import numpy as np

from tqdm import tqdm
from typing import List
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import config
from src.core.ontology import skill_ontology

# ============================= MODEL INITIALIZATION ============================= #
try:
    print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}...")
    GLOBAL_MODEL = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    print("(V) Embedding model loaded successfully.")
except Exception as e:
    print(f"(X) Error loading embedding model: {e}")
    GLOBAL_MODEL = None

# ============================= NLTK SETUP ============================= #
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Ensure required corpora are available
for corpus in ["stopwords", "wordnet"]:
    try:
        nltk.data.find(f"corpora/{corpus}")
    except LookupError:
        nltk.download(corpus)

lemmatizer = WordNetLemmatizer()
english_stopwords = set(stopwords.words("english"))

# ============================= EMBEDDING FUNCTIONS ============================= #
def generate_single_embedding(text: str) -> np.ndarray:
    """
    Convert a text string into a 768-dimensional embedding vector.

    Args:
        text (str): Input text.

    Returns:
        np.ndarray: Embedding vector or empty array if model/input invalid.
    """
    if GLOBAL_MODEL is None or not text or not isinstance(text, str):
        print("(W) Model not loaded or invalid input.")
        return np.array([])

    try:
        embeddings = GLOBAL_MODEL.encode([text], convert_to_tensor=False, show_progress_bar=False)
        return embeddings[0]
    except Exception as e:
        print(f"(X) Error during encoding: {e}")
        return np.array([])
    
def process_summaries_to_embeddings(summary_folder: str) -> dict:
    """
    Read .summary files, generate embeddings, and store them in a dictionary.

    Args:
        summary_folder (str): Path to folder containing .summary files.

    Returns:
        dict: Mapping of filename → embedding vector.
    """
    if GLOBAL_MODEL is None:
        print("⚠️ Cannot run, embedding model not loaded.")
        return {}

    all_embeddings = {}
    summary_files = [f for f in os.listdir(summary_folder) if f.endswith(".summary")]

    print(f"\nFound {len(summary_files)} summary files in {summary_folder}. Starting embedding...")

    for filename in tqdm(summary_files):
        filepath = os.path.join(summary_folder, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                summary_text = f.read()

            embedding_vector = generate_single_embedding(summary_text)
            if embedding_vector.size > 0:
                all_embeddings[filename] = embedding_vector
            else:
                print(f"Skipped {filename}: Could not generate vector.")
        except Exception as e:
            print(f"(X) Error processing {filename}: {e}")
            all_embeddings[filename] = None

    print("\n(V) Embedding process complete.")
    return all_embeddings

# ============================= FAISS INDEXING ============================= #
def build_rich_faiss_index_offline(summary_folder: str, json_folder: str, pdf_folder: str,
                                   output_folder: str, dimension: int) -> bool:
    """
    Build FAISS index and rich metadata mapping (summary, PDF, JSON).

    Args:
        summary_folder (str): Path to summary files.
        json_folder (str): Path to JSON metadata files.
        pdf_folder (str): Path to PDF files.
        output_folder (str): Path to save index and map.
        dimension (int): Expected embedding dimension.

    Returns:
        bool: True if successful, False otherwise.
    """
    if GLOBAL_MODEL is None:
        print("(X) Embedding model not loaded. Cannot build index.")
        return False

    all_vectors, rich_metadata = [], []
    summary_files = [f for f in os.listdir(summary_folder) if f.endswith(".summary")]

    for filename in tqdm(summary_files, desc="Indexing"):
        base_name = os.path.splitext(filename)[0]
        summary_path = os.path.join(summary_folder, filename)
        json_path = os.path.join(json_folder, f"{base_name}.json")
        pdf_path = os.path.join(pdf_folder, f"{base_name}.pdf")

        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                text = f.read()
            vector = generate_single_embedding(text)

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

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
                "tech_stack": data.get("tech_stack", {}),
            }

            rich_metadata.append(meta)
            all_vectors.append(vector)
        except Exception as e:
            print(f"⚠️ Skipped {filename}: {e}")

    if not all_vectors:
        print("(X) No valid vectors generated. Indexing aborted.")
        return False

    vectors_array = np.array(all_vectors).astype("float32")
    if len(vectors_array.shape) != 2 or vectors_array.shape[1] != dimension:
        print(f"(X) Incorrect vector shape {vectors_array.shape}, expected (N, {dimension}).")
        return False

    print(f"Batch embedding complete. Building FAISS index with {vectors_array.shape[0]} vectors...")

    faiss.normalize_L2(vectors_array)
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors_array)

    index_path = os.path.join(output_folder, config.FAISS_INDEX_FILE)
    map_path = os.path.join(output_folder, config.ID_MAP_FILE)

    faiss.write_index(index, index_path)
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(rich_metadata, f, indent=4)

    print(f"(V) FAISS Index saved to: {index_path}")
    print(f"(V) Rich ID Map saved to: {map_path}")
    return True

# ============================= TEXT PROCESSING ============================= #
def preprocess_text(text: str) -> List[str]:
    """
    Preprocess text for BM25:
    - Lowercase
    - Remove links and non-alphanumeric chars
    - Lemmatize tokens
    - Remove stopwords and blacklist terms
    """
    if not text or not isinstance(text, str):
        return []

    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9]", " ", text)

    tokens = text.split()
    processed_tokens = []

    blacklist = {"frid", "dto", "consequ", "actiontypes", "uct", "dee", "ieren", "cid"}

    for token in tokens:
        if len(token) < 2 or len(token) > 20:
            continue
        if token in english_stopwords or token in blacklist:
            continue
        if "ieren" in token or "uct" in token:
            continue

        lemma = lemmatizer.lemmatize(token)
        if len(lemma) > 1:
            processed_tokens.append(lemma)

    return processed_tokens

def flatten_tech_stack(json_data: dict) -> List[str]:
    """
    Flatten tech stack into a normalized skill list using ontology.
    """
    tech_stack = json_data.get("tech_stack", {})
    all_skills = []

    for category in tech_stack.values():
        if isinstance(category, list):
            for skill in category:
                all_skills.append(skill_ontology.normalize_skill(skill))

    languages = json_data.get("languages", [])
    for lang in languages:
        certs = lang.get("certifications_and_equivalents")
        if certs and isinstance(certs, list):
            all_skills.extend(certs)

    return list(set(all_skills))

def tokenize_with_skills(text: str, skills: list = None) -> List[str]:
    """
    Tokenize text and append normalized skills (including n-grams).
    
    Args:
        text (str): Input text.
        skills (list): Optional list of skills to inject into token set.
    
    Returns:
        List[str]: Tokenized corpus with skills included.
    """
    tokens = preprocess_text(text)

    if skills:
        for s in skills:
            if not s or not isinstance(s, str):
                continue

            s_clean = s.lower().strip()

            # Handle multi-word skills (convert to underscore form)
            if " " in s_clean:
                s_joined = re.sub(r"\s+", "_", s_clean)
                tokens.append(s_joined)

            # Add normalized alphanumeric form
            s_norm = re.sub(r"[^a-z0-9]", "", s_clean)
            if len(s_norm) > 1:
                tokens.append(s_norm)

    return tokens

def build_bm25_index(map_path: str, bm25_index_path: str):
    """
    Build BM25 index from rich metadata map.
    
    Args:
        map_path (str): Path to JSON metadata map file.
        bm25_index_path (str): Path to save BM25 index storage.
    
    Returns:
        Tuple[BM25Okapi, list]: BM25 index and rich metadata list.
    """
    print(f"Starting BM25 index build from: {map_path}")
    try:
        with open(map_path, "r", encoding="utf-8") as f:
            rich_metadata = json.load(f)
    except Exception as e:
        print(f"(X) Error loading Rich Metadata Map: {e}")
        return None, None

    tokenized_corpus = []

    for i, item in enumerate(rich_metadata):
        item["doc_id"] = i
        raw_text = item.get("summary_content", "")
        skills = item.get("skills", [])

        processed_tokens = tokenize_with_skills(raw_text, skills=skills)
        tokenized_corpus.append(processed_tokens)

    if not tokenized_corpus:
        print("(X) Error: Corpus is empty.")
        return None, None

    bm25 = BM25Okapi(tokenized_corpus)
    print(f"(V) BM25 Index built successfully for {len(tokenized_corpus)} documents.")

    storage_data = {
        "tokenized_corpus": tokenized_corpus,
        "rich_metadata_with_id": rich_metadata,
    }

    with open(bm25_index_path, "w", encoding="utf-8") as f:
        json.dump(storage_data, f, indent=2)

    print(f"(V) BM25 Corpus saved to: {bm25_index_path}")
    return bm25, rich_metadata