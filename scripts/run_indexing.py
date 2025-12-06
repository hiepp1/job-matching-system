import os
import sys

# ============================= PATH ============================= #
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import config
from src.data_pipeline.loader import process_cvs_to_json, generate_summaries
from src.data_pipeline.indexing import build_rich_faiss_index_offline, build_bm25_index

# ============================= MAIN PIPELINE ============================= #
def main():
    """
    End-to-end indexing pipeline:
    1. Convert CV PDFs → JSON
    2. Generate summaries from JSON
    3. Build FAISS index
    4. Build BM25 index
    """
    print("\n--- STARTING INDEXING PIPELINE ---")

    # Ensure required folders exist
    os.makedirs(config.CV_JSON_FOLDER, exist_ok=True)
    os.makedirs(config.CV_SUMMARY_FOLDER, exist_ok=True)
    os.makedirs(config.CV_DATASET_FOLDER, exist_ok=True)
    print("(V) Verified/created artifact folders.")

    # Step 1: PDF → JSON
    print("\n[Step 1/4] Processing CV PDFs into JSON...")
    process_cvs_to_json(
        cv_folder_path=config.CV_FOLDER,
        json_output_folder=config.CV_JSON_FOLDER,
    )
    print("✅ Step 1 complete.")

    # Step 2: JSON → Summaries
    print("\n[Step 2/4] Generating summaries from JSON...")
    generate_summaries(
        json_input_folder=config.CV_JSON_FOLDER,
        summary_output_folder=config.CV_SUMMARY_FOLDER,
    )
    print("✅ Step 2 complete.")

    # Step 3: Build FAISS Index
    print("\n[Step 3/4] Building FAISS Index...")
    build_rich_faiss_index_offline(
        summary_folder=config.CV_SUMMARY_FOLDER,
        json_folder=config.CV_JSON_FOLDER,
        pdf_folder=config.CV_FOLDER,
        output_folder=config.CV_DATASET_FOLDER,
        dimension=config.EMBEDDING_DIMENSION,
    )
    print("✅ Step 3 complete.")

    # Step 4: Build BM25 Index
    print("\n[Step 4/4] Building BM25 Index...")
    build_bm25_index(
        map_path=config.MAP_PATH,
        bm25_index_path=config.BM25_INDEX_PATH,
    )
    print("✅ Step 4 complete.")

    print("\n--- INDEXING PIPELINE COMPLETED ---")

if __name__ == "__main__":
    main()