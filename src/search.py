import faiss
import json
import numpy as np
import math
import re, string
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from typing import List

# Import các thành phần cần thiết
import config
from .indexing import GLOBAL_MODEL, generate_single_embedding, preprocess_text, lemmatizer, english_stopwords

# --- KHỞI TẠO CROSS-ENCODER  --- #
try:
    cross_encoder = CrossEncoder(config.CROSS_ENCODER_MODEL)
except Exception as e:
    print("Error loading cross encoder:", e)
    cross_encoder = None

# Bảng quy đổi CEFR ra điểm số để so sánh toán học
CEFR_MAP = {
    "null": 0, "unknown": 0,
    "a1": 1, "a2": 2,
    "b1": 3, "b2": 4,
    "c1": 5, "c2": 6, "native": 7
}

## ============== MAIN CODE =============== #
def load_faiss_index():
    try:
        index = faiss.read_index(config.INDEX_PATH)
        with open(config.MAP_PATH, 'r', encoding='utf-8') as f:
            rich_metadata = json.load(f)
        return index, rich_metadata
    except Exception as e:
        print(f"Lỗi khi tải FAISS Index: {e}")
        return None, None
    
def load_bm25_index():
    try:
        with open(config.BM25_INDEX_PATH, 'r', encoding='utf-8') as f:
            storage_data = json.load(f)
        tokenized_corpus = storage_data["tokenized_corpus"]
        rich_metadata = storage_data["rich_metadata_with_id"]
        bm25 = BM25Okapi(tokenized_corpus)
        return bm25, rich_metadata
    except Exception as e:
        print(f"Lỗi khi tải BM25 Index: {e}. Vui lòng chạy lại build_bm25_index().")
        return None, None

def search_faiss_index(index, query_vector, k: int = 5):

    query_vector = query_vector.reshape(1, -1).astype('float32')
    faiss.normalize_L2(query_vector)
    D, I = index.search(query_vector, k)

    results = []
    for rank, (score, index_id) in enumerate(zip(D[0], I[0])):
        if index_id >= 0:
             results.append({"index_id": int(index_id), "score": float(score), "rank": rank + 1})
    return results

def search_bm25(bm25_index, query: str, k: int = 10):
    tokenized_query = preprocess_text(query)
    if not tokenized_query: return []
    doc_scores = bm25_index.get_scores(tokenized_query)
    top_n_indices = np.argsort(doc_scores)[::-1]

    results = []
    for rank, index_id in enumerate(top_n_indices[:k]):
        score = doc_scores[index_id]
        if score > 0:
            results.append({"index_id": int(index_id), "bm25_score": float(score), "rank": rank + 1})
    return results

def compute_skill_overlap(jd_skills: List[str], cv_skills: List[str]) -> float:
    """Compute Jaccard-like overlap on normalized skill tokens. Returns 0-1."""
    if not jd_skills or not cv_skills:
        return 0.0
    # normalize tokens
    def norm_list(lst):
        return set([re.sub(r'[^a-z0-9]', '', s.lower()).strip() for s in lst if s])
    jd_set = norm_list(jd_skills)
    cv_set = norm_list(cv_skills)
    if not jd_set:
        return 0.0
    inter = jd_set & cv_set
    union = jd_set | cv_set
    return len(inter) / len(jd_set)

def generate_single_embedding(text: str) -> np.ndarray:
    if GLOBAL_MODEL is None or not text: return np.array([])
    embeddings = GLOBAL_MODEL.encode([text], convert_to_tensor=False, show_progress_bar=False)
    return embeddings[0]

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    processed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in english_stopwords]
    return processed_tokens

def cross_encoder_rerank(query: str, candidates: List[dict], top_k=10):
    """
    candidates: list of dicts with keys 'pdf_path' or 'summary_snippet' (text to compare)
    returns candidates sorted by cross-encoder score
    """
    if cross_encoder is None or not candidates:
        return candidates
    pairs = []
    for c in candidates:
        # prefer using structured summary or summary_snippet
        text = c.get("summary_snippet") or ""
        pairs.append((query, text))
    scores = cross_encoder.predict(pairs)
    for i, c in enumerate(candidates):
        c["cross_score"] = float(scores[i])
    return sorted(candidates, key=lambda x: x["cross_score"], reverse=True)[:top_k]

def normalize_list(scores):
    if not scores:
        return []
    mn, mx = min(scores), max(scores)
    if math.isclose(mx, mn):
        return [0.5 for _ in scores]
    return [(s - mn) / (mx - mn) for s in scores]
def compute_title_match(jd_title: str, cv_titles: List[str]) -> float:
    """So sánh chức danh JD với danh sách chức danh CV đã được LLM chuẩn hóa."""
    if not jd_title or not cv_titles:
        return 0.0

    # Chuẩn hóa JD title
    q = jd_title.lower().strip()

    # Chuẩn hóa CV titles
    c_titles = [t.lower().strip() for t in cv_titles]

    if q in c_titles:
        return 1.0 # Khớp hoàn hảo

    # khớp một phần (ví dụ: "Senior Software Engineer" vs "Software Engineer")
    for ct in c_titles:
        if q in ct or ct in q:
            return 0.5
    return 0.0

def compute_final_scores(faiss_results, bm25_results, rich_metadata, jd_struct: dict, weights=None, top_k=50):
    """
    Kết hợp nhiều tín hiệu (đã được LLM trích xuất) vào điểm cuối cùng.
    """
    if weights is None:
        weights = {
            "semantic": 0.50,
            "bm25": 0.10,
            "skill": 0.30,
            "title": 0.00,
            "industry": 0.30
        }

    # 1. Thu thập điểm thô
    doc_scores = {}
    for r in faiss_results:
        doc_id = int(r["index_id"])
        doc_scores.setdefault(doc_id, {})["semantic"] = r["score"]
    for r in bm25_results:
        doc_id = int(r["index_id"])
        doc_scores.setdefault(doc_id, {})["bm25"] = r["bm25_score"]

    # 2. Chuẩn hóa điểm (Normalization)
    semantic_vals = [v.get("semantic", 0.0) for v in doc_scores.values()]
    bm25_vals = [v.get("bm25", 0.0) for v in doc_scores.values()]
    semantic_norm = normalize_list(semantic_vals)
    bm25_norm = normalize_list(bm25_vals)

    doc_ids = list(doc_scores.keys())
    for i, doc_id in enumerate(doc_ids):
        doc_scores[doc_id]["semantic_norm"] = semantic_norm[i]
        doc_scores[doc_id]["bm25_norm"] = bm25_norm[i]

    # 3. Lấy dữ liệu thông minh từ JD (đã được LLM trích xuất)
    jd_skills_dict = jd_struct.get("skills", {})
    if isinstance(jd_skills_dict, dict):
        req_skills = jd_skills_dict.get("required", [])
        pref_skills = jd_skills_dict.get("preferred", [])
        jd_skills = req_skills + pref_skills 
    else:
        # Fallback cho cấu trúc cũ (nếu có)
        jd_skills = jd_skills_dict if isinstance(jd_skills_dict, list) else []

    job_info = jd_struct.get("job_info", {})
    jd_title = job_info.get("job_title", jd_struct.get("job", ""))
    jd_industry = job_info.get("domain", jd_struct.get("detected_industry", "unknown"))

    job_info = jd_struct.get("job_info", {})
    jd_min_yoe = job_info.get("min_years_experience", 0)
    jd_level = job_info.get("seniority_level", "").lower()

    # 4. Tính điểm cho từng CV
    doc_ids = list(doc_scores.keys())
    
    for doc_id in doc_ids:
        md = rich_metadata[doc_id]

        cv_yoe = md.get("yoe", 0)
        cv_level = md.get("seniority", "").lower()
        print(f"JD min YOE: {jd_min_yoe}, CV YOE: {cv_yoe}, JD level: {jd_level}, CV level: {cv_level}")

        seniority_factor = calculate_seniority_penalty(jd_min_yoe, cv_yoe, jd_level, cv_level)

        cv_skills = md.get("skills", [])
        cv_titles = [md.get("role", "")]
        cv_industry = md.get("industry", "unknown")

        skill_score = compute_skill_overlap(jd_skills, cv_skills)
        title_score = compute_title_match(jd_title, cv_titles)
        
        industry_match = 0.0 
        
        doc_scores[doc_id]["skill_score"] = skill_score
        doc_scores[doc_id]["title_score"] = title_score
        doc_scores[doc_id]["industry_match"] = industry_match

        base = (weights["semantic"] * doc_scores[doc_id]["semantic_norm"]
                + weights["bm25"] * doc_scores[doc_id]["bm25_norm"]
                + weights["skill"] * skill_score
                + weights["title"] * title_score)

        final = (base * (1.0 + weights["industry"] * industry_match)) * seniority_factor
        doc_scores[doc_id]["final_score"] = final
        doc_scores[doc_id]["seniority_factor"] = seniority_factor
    
    # 6. Trả về kết quả đã sắp xếp
    sorted_docs = sorted(doc_scores.items(), key=lambda kv: kv[1]["final_score"], reverse=True)
    results = []
    for rank, (doc_id, scores) in enumerate(sorted_docs[:top_k]):
        md = rich_metadata[doc_id]

        cv_skills = md.get("skills", [])
        results.append({
            "rank": rank + 1,
            "document_id": doc_id,
            "final_score": scores["final_score"],
            "semantic": scores.get("semantic_norm", 0.0),
            "bm25": scores.get("bm25_norm", 0.0),
            "skill": scores.get("skill_score", 0.0),
            "title": scores.get("title_score", 0.0),
            "industry_match": scores.get("industry_match", 0.0),
            "summary_file": md.get("summary_filename"),
            "pdf_path": md.get("pdf_filepath"),
            "summary_snippet": md.get("summary_content", "")[:800],
            "cv_skills_list": cv_skills
        })
    return results

def hybrid_search_v2(jd_query: str, jd_struct: dict = None, k_faiss=50, k_bm25=100, top_show=10):
    """
    jd_struct: structured JD dict (if you have parsed json from Gemini)
    """
    faiss_index, faiss_metadata = load_faiss_index()
    bm25_index, bm25_metadata = load_bm25_index()
    if faiss_index is None or bm25_index is None:
        print("Index missing")
        return []

    query_vec = generate_single_embedding(jd_query)
    faiss_results = search_faiss_index(faiss_index, query_vec, k=k_faiss)
    bm25_results = search_bm25(bm25_index, jd_query, k=k_bm25)

    # Compute composite final scores
    final_candidates = compute_final_scores(faiss_results, bm25_results, faiss_metadata, jd_struct, weights=None, top_k=top_show*4)

    # rerank top candidates with cross-encoder (optional)
    reranked = cross_encoder_rerank(jd_query, final_candidates, top_k=top_show)

    return final_candidates


## ========================= PENALTY ==================== #
def calculate_english_penalty(jd_cefr_str, cv_cefr_str):
    """So sánh dựa trên thang chuẩn CEFR"""
    jd_score = CEFR_MAP.get(str(jd_cefr_str).lower(), 0)
    cv_score = CEFR_MAP.get(str(cv_cefr_str).lower(), 0)
    
    if jd_score == 0: return 1.0
    if cv_score >= jd_score: return 1.0 
    
    if jd_score - cv_score == 1: return 0.8
    
    return 0.4

def calculate_location_penalty(jd_loc, cv_loc):
    """So sánh địa điểm đã chuẩn hóa"""
    if not jd_loc or not cv_loc: return 1.0
    
    j_clean = jd_loc.strip().lower()
    c_clean = cv_loc.strip().lower()
    
    if j_clean == c_clean: return 1.0
    
    return 0.5

def calculate_seniority_penalty(jd_min_yoe, cv_yoe, jd_level, cv_level):
    """
    Tính hệ số phạt dựa trên chênh lệch kinh nghiệm (Phiên bản Logic: Research-Grade).
    """
    # 1. Nếu JD không yêu cầu kinh nghiệm hoặc yêu cầu = 0 -> Không phạt
    if jd_min_yoe is None or jd_min_yoe <= 0:
        return 1.0

    # 2. Xử lý dữ liệu thiếu
    if cv_yoe is None:
        cv_yoe = 0.0

    # 3. Nếu CV dư hoặc đủ kinh nghiệm -> Tốt (1.0)
    if cv_yoe >= jd_min_yoe:
        return 1.0
    
    gap = jd_min_yoe - cv_yoe

    # TRƯỜNG HỢP 1: JD "Senior" (Yêu cầu >= 3 năm)
    if jd_min_yoe >= 3.0 and gap > 2.0:
        return 0.1 

    # TRƯỜNG HỢP 2: JD "Junior" (Yêu cầu <= 2 năm)
    if jd_min_yoe <= 2.0:
        return 0.85 

    # TRƯỜNG HỢP 3: Các trường hợp còn lại (Middle)
    ratio = cv_yoe / jd_min_yoe
    return max(0.4, ratio) 
