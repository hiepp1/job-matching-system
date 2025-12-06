import json
import math
import os
import string
from typing import List, Tuple, Dict, Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

import config
from ..data_pipeline.indexing import (
    GLOBAL_MODEL,
    generate_single_embedding,
    preprocess_text,
    lemmatizer,
    english_stopwords,
)
from src.core.ontology import skill_ontology

# ============================= MODEL INITIALIZATION ============================= #
try:
    cross_encoder = CrossEncoder(config.CROSS_ENCODER_MODEL)
except Exception as e:
    print("(X) Warning: Error loading cross encoder:", e)
    cross_encoder = None

# ============================= CONSTANTS ============================= #
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
    "unknown": 0,
}

CEFR_MAP = {
    "a1": 1,
    "a2": 2,
    "b1": 3,
    "b2": 4,
    "c1": 5,
    "c2": 6,
    "unknown": 0,
}

# ============================= INDEX LOADING ============================= #
def load_faiss_index() -> Tuple[Any, Dict[int, dict]]:
    """
    Load FAISS index and associated rich metadata.

    Returns:
        Tuple[index, rich_metadata]: FAISS index and metadata mapping.
        (None, None) if load fails.
    """
    try:
        index = faiss.read_index(config.INDEX_PATH)
        with open(config.MAP_PATH, "r", encoding="utf-8") as f:
            rich_metadata = json.load(f)
        return index, rich_metadata
    except Exception as e:
        print(f"(X) Error loading FAISS Index: {e}")
        return None, None
    
def load_bm25_index() -> Tuple[BM25Okapi, Dict[int, dict]]:
    """
    Load BM25 index and associated rich metadata.

    Returns:
        Tuple[bm25, rich_metadata]: BM25Okapi instance and metadata mapping.
        (None, None) if load fails.
    """
    try:
        with open(config.BM25_INDEX_PATH, "r", encoding="utf-8") as f:
            storage_data = json.load(f)

        tokenized_corpus = storage_data["tokenized_corpus"]
        rich_metadata = storage_data["rich_metadata_with_id"]
        bm25 = BM25Okapi(tokenized_corpus)
        return bm25, rich_metadata
    except Exception as e:
        print(f"(X) Error loading BM25 Index: {e}. Please rebuild via build_bm25_index().")
        return None, None

# ============================= SEARCH ============================= #
def search_faiss_index(index: Any, query_vector: np.ndarray, k: int = 5) -> List[dict]:
    """
    Search FAISS index with a query vector.

    Args:
        index: FAISS index instance.
        query_vector: Embedding vector for the query.
        k: Number of nearest neighbors to retrieve.

    Returns:
        List of search results with index_id, score, and rank.
    """
    query_vector = query_vector.reshape(1, -1).astype("float32")
    faiss.normalize_L2(query_vector)
    D, I = index.search(query_vector, k)

    results = []
    for rank, (score, index_id) in enumerate(zip(D[0], I[0])):
        if index_id >= 0:
            results.append({"index_id": int(index_id), "score": float(score), "rank": rank + 1})
    return results

def search_bm25(bm25_index: BM25Okapi, query: str, k: int = 10) -> List[dict]:
    """
    Search BM25 index using the tokenized query.

    Args:
        bm25_index: BM25Okapi instance.
        query: Query text.
        k: Number of top results.

    Returns:
        List of results with index_id, bm25_score, and rank.
    """
    tokenized_query = preprocess_text(query)
    if not tokenized_query:
        return []

    doc_scores = bm25_index.get_scores(tokenized_query)
    top_n_indices = np.argsort(doc_scores)[::-1]

    results = []
    for rank, index_id in enumerate(top_n_indices[:k]):
        score = doc_scores[index_id]
        if score > 0:
            results.append({"index_id": int(index_id), "bm25_score": float(score), "rank": rank + 1})
    return results

# ============================= SKILL & TITLE MATCHING ============================= #
def compute_skill_overlap(jd_skills: list, cv_skills: list) -> float:
    """
    Compute skill match score using the Skill Ontology.

    Rules:
    - Exact canonical match: 1.0
    - Complementary skills: 0.6
    - Alternative skills: 0.4
    - Related skills: via ontology relationships

    Args:
        jd_skills: List of JD required/preferred skills.
        cv_skills: List of candidate skills.

    Returns:
        Float in [0.0, 1.0] representing overlap score.
    """
    if not jd_skills or not cv_skills:
        return 0.0

    jd_norm = [skill_ontology.normalize_skill(s) for s in jd_skills]
    cv_norm = [skill_ontology.normalize_skill(s) for s in cv_skills]
    cv_set = {s.lower() for s in cv_norm}

    total_score = 0.0

    for jd_skill in jd_norm:
        jd_s_lower = jd_skill.lower()

        # Exact match
        if jd_s_lower in cv_set:
            total_score += 1.0
            continue

        # Relationship match (best fuzzy score)
        best_rel_score = 0.0
        for cv_skill in cv_set:
            rel_score = skill_ontology.check_relationship(jd_skill, cv_skill)
            if rel_score > best_rel_score:
                best_rel_score = rel_score

        total_score += best_rel_score

    final_score = total_score / len(jd_skills)
    return min(1.0, final_score)

def compute_title_match(jd_title: str, cv_titles: List[str]) -> float:
    """
    Compare JD title with normalized CV titles.

    Exact match -> 1.0
    Partial match (substring containment either way) -> 0.5
    No match -> 0.0
    """
    if not jd_title or not cv_titles:
        return 0.0

    q = jd_title.lower().strip()
    c_titles = [t.lower().strip() for t in cv_titles]

    if q in c_titles:
        return 1.0

    for ct in c_titles:
        if q in ct or ct in q:
            return 0.5

    return 0.0

# ============================= SCORING ============================= #
def normalize_list(scores: List[float]) -> List[float]:
    """
    Normalize a list of scores to [0, 1]. If all equal, return 0.5 for each.
    """
    if not scores:
        return []
    mn, mx = min(scores), max(scores)
    if math.isclose(mx, mn):
        return [0.5 for _ in scores]
    return [(s - mn) / (mx - mn) for s in scores]

def compute_final_scores(
    faiss_results: List[dict],
    bm25_results: List[dict],
    rich_metadata: Dict[int, dict],
    jd_struct: dict,
    weights: dict = None,
    top_k: int = 50,
) -> List[dict]:
    """
    Combine multiple signals (semantic, lexical, skill, title) into a final score.

    Args:
        faiss_results: Results from FAISS semantic search.
        bm25_results: Results from BM25 lexical search.
        rich_metadata: Document metadata mapping.
        jd_struct: Parsed JD structure (skills, job_info, etc.).
        weights: Weight configuration for signals.
        top_k: Number of top results to return.

    Returns:
        Ranked list of candidates with detailed scoring breakdown.
    """
    if weights is None:
        weights = {
            "semantic": 0.50,
            "bm25": 0.10,
            "skill": 0.25,
            "title": 0.00,
            "industry": 0.30,
        }

    # 1. Aggregate raw scores
    doc_scores: Dict[int, dict] = {}
    for r in faiss_results:
        doc_id = int(r["index_id"])
        doc_scores.setdefault(doc_id, {})["semantic"] = r["score"]

    for r in bm25_results:
        doc_id = int(r["index_id"])
        doc_scores.setdefault(doc_id, {})["bm25"] = r["bm25_score"]

    # 2. Normalize
    semantic_vals = [v.get("semantic", 0.0) for v in doc_scores.values()]
    bm25_vals = [v.get("bm25", 0.0) for v in doc_scores.values()]
    semantic_norm = normalize_list(semantic_vals)
    bm25_norm = normalize_list(bm25_vals)

    doc_ids = list(doc_scores.keys())
    for i, doc_id in enumerate(doc_ids):
        doc_scores[doc_id]["semantic_norm"] = semantic_norm[i]
        doc_scores[doc_id]["bm25_norm"] = bm25_norm[i]

    # 3. Extract JD data
    jd_skills_dict = jd_struct.get("skills", {})
    if isinstance(jd_skills_dict, dict):
        req_skills = jd_skills_dict.get("required", [])
        pref_skills = jd_skills_dict.get("preferred", [])
        jd_skills = req_skills + pref_skills
    else:
        jd_skills = jd_skills_dict if isinstance(jd_skills_dict, list) else []

    job_info = jd_struct.get("job_info", {})
    jd_cefr = job_info.get("min_english_cefr", {})
    jd_title = job_info.get("job_title", jd_struct.get("job", ""))
    jd_min_yoe = job_info.get("min_years_experience", 0)
    jd_level = job_info.get("seniority_level", "").lower()
    jd_industry = job_info.get("domain", jd_struct.get("detected_industry", "unknown"))

    # 4. Score each candidate
    for doc_id in doc_ids:
        md = rich_metadata[doc_id]

        cv_yoe = md.get("yoe", 0)
        cv_level = md.get("seniority", "").lower()
        print(f"JD min YOE: {jd_min_yoe}, CV YOE: {cv_yoe}, JD level: {jd_level}, CV level: {cv_level}")

        # Penalties and bonuses
        p_seniority = calculate_seniority_penalty(jd_min_yoe, md.get("yoe"), jd_level, md.get("seniority"))
        p_english = calculate_english_penalty(jd_cefr, md.get("english_level"))
        bonus_achievement = calculate_achievement_bonus(md.get("certifications"), md.get("awards"))

        cv_skills = md.get("skills", [])
        cv_titles = [md.get("role", "")]
        cv_industry = md.get("industry", "unknown")

        skill_score = compute_skill_overlap(jd_skills, cv_skills)
        title_score = compute_title_match(jd_title, cv_titles)
        industry_match = 0.0  # Placeholder if you add industry matching later

        doc_scores[doc_id]["skill_score"] = skill_score
        doc_scores[doc_id]["title_score"] = title_score
        doc_scores[doc_id]["industry_match"] = industry_match

        base = (
            weights["semantic"] * doc_scores[doc_id]["semantic_norm"]
            + weights["bm25"] * doc_scores[doc_id]["bm25_norm"]
            + weights["skill"] * skill_score
            + weights["title"] * title_score
        )

        final = (base * (1.0 + weights["industry"] * industry_match)) * p_seniority * p_english * bonus_achievement

        doc_scores[doc_id]["final_score"] = final
        doc_scores[doc_id]["factors"] = {
            "sen_penalty": p_seniority,
            "eng_penalty": p_english,
            "ach_bonus": bonus_achievement,
        }

    # 5. Collect results
    sorted_docs = sorted(doc_scores.items(), key=lambda kv: kv[1]["final_score"], reverse=True)
    results = []
    for rank, (doc_id, scores) in enumerate(sorted_docs[:top_k]):
        md = rich_metadata[doc_id]
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
            "cv_skills_list": md.get("skills", []),
        })

    return results

# ============================= PIPELINE ============================= #
def hybrid_search_v2(jd_query: str, jd_struct: dict = None, k_faiss: int = 50, k_bm25: int = 100, top_show: int = 10) -> List[dict]:
    """
    Hybrid retrieval + scoring pipeline:
    - FAISS semantic search
    - BM25 lexical search
    - Composite scoring (semantic, lexical, skills, title)
    - Optional cross-encoder reranking (scores attached if available)

    Args:
        jd_query: Raw JD text query.
        jd_struct: Parsed JD JSON (if available).
        k_faiss: Number of FAISS candidates.
        k_bm25: Number of BM25 candidates.
        top_show: Number of final results to show.

    Returns:
        Final candidate list with detailed scores.
    """
    faiss_index, faiss_metadata = load_faiss_index()
    bm25_index, bm25_metadata = load_bm25_index()

    if faiss_index is None or bm25_index is None:
        print("Index missing.")
        return []

    query_vec = generate_single_embedding(jd_query)
    faiss_results = search_faiss_index(faiss_index, query_vec, k=k_faiss)
    bm25_results = search_bm25(bm25_index, jd_query, k=k_bm25)

    final_candidates = compute_final_scores(
        faiss_results,
        bm25_results,
        faiss_metadata,
        jd_struct,
        weights=None,
        top_k=top_show * 2
    )

    # Optional cross-encoder reranking
    reranked = cross_encoder_rerank(jd_query, final_candidates, top_k=top_show)

    # Return original final_candidates to keep same logic/return signature
    return final_candidates

# ============================= PENALTIES & BONUSES ============================= #
def calculate_seniority_penalty(jd_min_yoe, cv_yoe, jd_level_str, cv_level_str) -> float:
    """
    Penalty based on years-of-experience gap and seniority level gap.

    Returns:
        Penalty multiplier in [0.1, 1.0].
    """
    if jd_min_yoe is None:
        jd_min_yoe = 0
    if cv_yoe is None:
        cv_yoe = 0

    jd_lvl_num = SENIORITY_MAP.get(str(jd_level_str).lower().split("/")[0].strip(), 0)
    cv_lvl_num = SENIORITY_MAP.get(str(cv_level_str).lower().strip(), 0)

    # No requirement -> no penalty
    if jd_min_yoe <= 0 and jd_lvl_num == 0:
        return 1.0

    # Years check
    if jd_min_yoe > 0:
        if jd_min_yoe >= 3.0 and cv_yoe < 1.0:
            return 0.1
        if cv_yoe < jd_min_yoe:
            ratio = cv_yoe / jd_min_yoe
            return max(0.5, ratio)

    # Level check
    if jd_lvl_num == 0 or cv_lvl_num == 0:
        return 1.0

    gap = jd_lvl_num - cv_lvl_num

    if gap > 0:
        if gap == 1:
            return 0.9
        if gap == 2:
            return 0.6
        if gap >= 3:
            return 0.2

    # Overqualified: mild penalty
    if gap < -2:
        return 0.85

    return 1.0

def calculate_english_penalty(jd_cefr, cv_cefr) -> float:
    """
    English level penalty based on CEFR comparison.

    Args:
        jd_cefr: JD requirement (e.g., "B2", "Native").
        cv_cefr: Candidate level (e.g., "A1", "C1").

    Returns:
        Penalty multiplier in [0.5, 1.0].
    """
    if not jd_cefr or str(jd_cefr).lower() in ["none", "null", "n/a", "unknown"]:
        return 1.0

    jd_score = CEFR_MAP.get(str(jd_cefr).lower().strip(), 0)
    cv_score = CEFR_MAP.get(str(cv_cefr).lower().strip(), 0)

    if jd_score <= 2:
        return 1.0
    if cv_score == 0:
        return 0.9
    if cv_score >= jd_score:
        return 1.0
    if jd_score - cv_score == 1:
        return 0.85
    return 0.5

def calculate_achievement_bonus(certs, awards) -> float:
    """
    Achievement bonus:
    - Certificates: +1% each (max +5%)
    - Awards: +3% each (max +15%)

    Returns:
        Bonus multiplier >= 1.0.
    """
    bonus = 1.0

    if certs and isinstance(certs, list):
        bonus += min(len(certs) * 0.01, 0.05)

    if awards and isinstance(awards, list):
        bonus += min(len(awards) * 0.03, 0.15)

    return bonus


## Additional Reranking with Cross-Encoder ##
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