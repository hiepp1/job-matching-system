import faiss
import json
import numpy as np
import math
import re, string
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from typing import List

import config

from ..data_pipeline.indexing import GLOBAL_MODEL, generate_single_embedding, preprocess_text, lemmatizer, english_stopwords
from src.core.ontology import skill_ontology

try:
    cross_encoder = CrossEncoder(config.CROSS_ENCODER_MODEL)
except Exception as e:
    print("Error loading cross encoder:", e)
    cross_encoder = None

# Bảng quy đổi seniority/cefr thành số để so sánh
SENIORITY_MAP = {
    "intern": 1,
    "fresher": 2,
    "junior": 3,
    "middle": 4,
    "senior": 5,
    "lead": 6,
    "manager": 7,
    "head": 7,
    "director": 8,
    "unknown": 0
}
CEFR_MAP = {
    "a1": 1,
    "a2": 2,
    "b1": 3,
    "b2": 4,
    "c1": 5,
    "c2": 6,
    "unknown": 0
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

def compute_skill_overlap(jd_skills: list, cv_skills: list) -> float:
    """
    Tính điểm khớp kỹ năng dựa trên Ontology (Knowledge Graph).
    - Match đúng tên (Canonical): 1.0 điểm
    - Match bổ trợ (Complement): 0.6 điểm
    - Match thay thế (Alternative): 0.4 điểm
    """
    if not jd_skills or not cv_skills:
        return 0.0

    jd_norm = [skill_ontology.normalize_skill(s) for s in jd_skills]
    cv_norm = [skill_ontology.normalize_skill(s) for s in cv_skills]
    
    cv_set = set([s.lower() for s in cv_norm])
    
    total_score = 0.0
    matched_skills = set() 
    
    for jd_skill in jd_norm:
        jd_s_lower = jd_skill.lower()
        
        # A. Kiểm tra khớp chính xác (Exact Match)
        if jd_s_lower in cv_set:
            total_score += 1.0
            matched_skills.add(jd_skill)
            continue 
            
        # B. Kiểm tra quan hệ (Relationship Match) - Fuzzy Logic
        best_rel_score = 0.0
        for cv_skill in cv_set:
            # Gọi hàm check trong Ontology Manager
            rel_score = skill_ontology.check_relationship(jd_skill, cv_skill)
            if rel_score > best_rel_score:
                best_rel_score = rel_score
        
        total_score += best_rel_score

    # Score = Tổng điểm đạt được / Tổng số skill JD yêu cầu
    final_score = total_score / len(jd_skills)
    return min(1.0, final_score)

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
            "skill": 0.25,
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

    # 3. Lấy dữ liệu thông minh từ JD
    jd_skills_dict = jd_struct.get("skills", {})
    if isinstance(jd_skills_dict, dict):
        req_skills = jd_skills_dict.get("required", [])
        pref_skills = jd_skills_dict.get("preferred", [])
        jd_skills = req_skills + pref_skills 
    else:
        # Fallback cho cấu trúc cũ (nếu có)
        jd_skills = jd_skills_dict if isinstance(jd_skills_dict, list) else []

    job_info = jd_struct.get("job_info", {})
    
    jd_cefr = job_info.get("min_english_cefr", {})
    jd_title = job_info.get("job_title", jd_struct.get("job", ""))
    jd_min_yoe = job_info.get("min_years_experience", 0)
    jd_level = job_info.get("seniority_level", "").lower()
    jd_industry = job_info.get("domain", jd_struct.get("detected_industry", "unknown"))

    # 4. Tính điểm cho từng CV
    doc_ids = list(doc_scores.keys())
    
    for doc_id in doc_ids:
        md = rich_metadata[doc_id]

        cv_yoe = md.get("yoe", 0)
        cv_level = md.get("seniority", "").lower()
        print(f"JD min YOE: {jd_min_yoe}, CV YOE: {cv_yoe}, JD level: {jd_level}, CV level: {cv_level}")

        #1. Senirity and English Penalty
        p_seniority = calculate_seniority_penalty(
            jd_min_yoe, md.get("yoe"), jd_level, md.get("seniority")
        )
        p_english = calculate_english_penalty(jd_cefr, md.get("english_level"))

        #2. Achievement Bonus
        bonus_achievement = calculate_achievement_bonus(
            md.get("certifications"), 
            md.get("awards")
        )

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

        final = (base * (1.0 + weights["industry"] * industry_match)) * p_seniority * p_english * bonus_achievement
        
        doc_scores[doc_id]["final_score"] = final
        doc_scores[doc_id]["factors"] = {
            "sen_penalty": p_seniority,
            "eng_penalty": p_english,
            "ach_bonus": bonus_achievement
        }
    
    # 6. Return top_k results
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
            "factors": scores.get("factors", {}),
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
    final_candidates = compute_final_scores(faiss_results, bm25_results, faiss_metadata, jd_struct, weights=None, top_k=top_show*2)

    # rerank top candidates with cross-encoder (optional)
    reranked = cross_encoder_rerank(jd_query, final_candidates, top_k=top_show)

    return final_candidates

## ========================= PENALTY ==================== #
def calculate_seniority_penalty(jd_min_yoe, cv_yoe, jd_level_str, cv_level_str):
    """
    Tính hệ số phạt dựa trên chênh lệch kinh nghiệm và cấp bậc.
    Trả về hệ số từ 0.1 (Phạt nặng) đến 1.0 (Không phạt).
    """
    # --- 1. XỬ LÝ DỮ LIỆU ĐẦU VÀO ---
    if jd_min_yoe is None: jd_min_yoe = 0
    if cv_yoe is None: cv_yoe = 0
    
    jd_lvl_num = SENIORITY_MAP.get(str(jd_level_str).lower().split('/')[0].strip(), 0)
    cv_lvl_num = SENIORITY_MAP.get(str(cv_level_str).lower().strip(), 0)

    # Nếu JD không yêu cầu gì đặc biệt -> Không phạt
    if jd_min_yoe <= 0 and jd_lvl_num == 0:
        return 1.0

    # --- 2. KIỂM TRA SỐ NĂM  ---
    # Luật: Thiếu năm kinh nghiệm nghiêm trọng -> Phạt nặng
    if jd_min_yoe > 0:
        # Case: JD Senior (3+) mà CV < 1 năm -> Phạt cực nặng
        if jd_min_yoe >= 3.0 and cv_yoe < 1.0:
            return 0.1
        
        # Case: Thiếu năm thường (Cần 3 có 2) -> Phạt theo tỷ lệ
        if cv_yoe < jd_min_yoe:
            ratio = cv_yoe / jd_min_yoe
            return max(0.5, ratio) # Giữ lại ít nhất 0.5 điểm

    # --- 3. KIỂM TRA CẤP BẬC (LEVEL CHECK) ---
    # Nếu không xác định được level của 1 trong 2 bên -> Bỏ qua check level, tin vào số năm
    if jd_lvl_num == 0 or cv_lvl_num == 0:
        return 1.0

    gap = jd_lvl_num - cv_lvl_num
    
    # A. CV thấp hơn JD (Underqualified)
    if gap > 0:
        if gap == 1: return 0.9  # JD Middle - CV Junior -> Chấp nhận được (0.9)
        if gap == 2: return 0.6  # JD Senior - CV Junior -> Hơi đuối (0.6)
        if gap >= 3: return 0.2  # JD Lead - CV Fresher -> Loại (0.2)
        
    # B. CV cao hơn JD (Overqualified) - Tùy chọn
    # Ví dụ: Tuyển Intern mà Senior nộp -> Có thể họ spam hoặc sẽ sớm nghỉ việc -> Phạt nhẹ
    if gap < -2: 
        return 0.85 

    return 1.0

def calculate_english_penalty(jd_cefr, cv_cefr):
    """
    Tính phạt tiếng Anh.
    - jd_cefr: Yêu cầu từ JD (VD: "B2", "Native").
    - cv_cefr: Trình độ CV (VD: "A1", "C1").
    """
    # 1. Nếu JD KHÔNG YÊU CẦU (Null hoặc rỗng) -> KHÔNG PHẠT (1.0)
    if not jd_cefr or str(jd_cefr).lower() in ["none", "null", "n/a", "unknown"]:
        return 1.0

    # 2. Lấy điểm số để so sánh
    jd_score = CEFR_MAP.get(str(jd_cefr).lower().strip(), 0)
    cv_score = CEFR_MAP.get(str(cv_cefr).lower().strip(), 0)
    
    # Nếu JD yêu cầu quá thấp (A1/A2) -> Coi như không phạt
    if jd_score <= 2: return 1.0
    
    # Nếu CV không có thông tin -> Phạt nhẹ (0.9)
    if cv_score == 0: return 0.9
    
    # Nếu CV >= JD -> Tốt (1.0)
    if cv_score >= jd_score: return 1.0
    
    # Nếu thiếu 1 bậc (VD: Cần B2 có B1) -> Phạt nhẹ (0.85)
    if jd_score - cv_score == 1: return 0.85
    
    # Nếu thiếu nhiều -> Phạt nặng (0.5)
    return 0.5

def calculate_achievement_bonus(certs, awards):
    """
    Tính điểm thưởng cho thành tích.
    - Certs: +1% mỗi cái (Max 5%)
    - Awards: +3% mỗi cái (Max 15%)
    """
    bonus = 1.0
    
    # Cộng điểm chứng chỉ (Học tập)
    if certs and isinstance(certs, list):
        bonus += min(len(certs) * 0.01, 0.05) 
        
    # Cộng điểm giải thưởng (Thành tích xuất sắc)
    if awards and isinstance(awards, list):
        bonus += min(len(awards) * 0.03, 0.15) 
        
    return bonus