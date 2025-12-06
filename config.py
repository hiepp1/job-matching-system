"""
Configuration file for CV/Job Description processing pipeline.
Defines paths, model settings, and artifact locations.
"""
import os

# ============================= BASE SETTINGS ============================= #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# External storage (Google Drive)
DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1Z9uFl_Pfz6ToYbFXJk7A6tHabkkusNRr"
)

TOP_N_TO_DISPLAY = 10
# ============================= MODEL CONFIG ============================= #
# Embedding model (SentenceTransformer)
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIMENSION = 768

# Cross-encoder model (fine-tuned or pretrained)
# CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_MODEL = os.path.join(BASE_DIR, "artifacts", "my_fine_tuned_model")

# ============================= DATA DIRECTORIES ============================= #
# Raw data folders
CV_FOLDER = os.path.join(BASE_DIR, "data", "cv_pdfs")
JD_FOLDER = os.path.join(BASE_DIR, "data", "jd_pdfs")

# Artifact folders (intermediate outputs)
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
CV_JSON_FOLDER = os.path.join(ARTIFACTS_DIR, "json_cv")
CV_SUMMARY_FOLDER = os.path.join(ARTIFACTS_DIR, "summary_cv")
CV_DATASET_FOLDER = os.path.join(ARTIFACTS_DIR, "cv_database")

# ============================= INDEX FILES ============================= #
FAISS_INDEX_FILE = "resume_summary_index.faiss"
ID_MAP_FILE = "faiss_id_map_rich.json"
BM25_INDEX_FILE = "bm25_tokenized_corpus.json"

INDEX_PATH = os.path.join(CV_DATASET_FOLDER, FAISS_INDEX_FILE)
MAP_PATH = os.path.join(CV_DATASET_FOLDER, ID_MAP_FILE)
BM25_INDEX_PATH = os.path.join(CV_DATASET_FOLDER, BM25_INDEX_FILE)
