import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import config 
from src.data_pipeline.loader import process_cvs_to_json, generate_summaries
from src.data_pipeline.indexing import build_rich_faiss_index_offline, build_bm25_index

def main():
    print("--- BẮT ĐẦU QUÁ TRÌNH INDEXING ---")

    # 1. Đảm bảo các thư mục tồn tại
    os.makedirs(config.CV_JSON_FOLDER, exist_ok=True)
    os.makedirs(config.CV_SUMMARY_FOLDER, exist_ok=True)
    os.makedirs(config.CV_DATASET_FOLDER, exist_ok=True)
    print("Đã kiểm tra/tạo các thư mục artifacts.")

    # 2. Step 1: PDF -> JSON
    print("\n[Bước 1/4] Đang xử lý CVs PDF sang JSON...")
    process_cvs_to_json(
        cv_folder_path=config.CV_FOLDER,
        json_output_folder=config.CV_JSON_FOLDER
    )
    print("Hoàn tất Bước 1.")

    # 3.  Step 2: JSON -> Summary
    print("\n[Bước 2/4] Đang tạo Summaries từ JSON...")
    generate_summaries(
        json_input_folder=config.CV_JSON_FOLDER,
        summary_output_folder=config.CV_SUMMARY_FOLDER
    )
    print("Hoàn tất Bước 2.")

    # 4.  Step 3: Tạo FAISS Index
    print("\n[Bước 3/4] Đang xây dựng FAISS Index...")
    build_rich_faiss_index_offline(
        summary_folder=config.CV_SUMMARY_FOLDER,
        json_folder=config.CV_JSON_FOLDER,
        pdf_folder=config.CV_FOLDER,
        output_folder=config.CV_DATASET_FOLDER,
        dimension=config.EMBEDDING_DIMENSION
    )
    print("Hoàn tất Bước 3.")

    # 5.  Step 4: Tạo BM25 Index
    print("\n[Bước 4/4] Đang xây dựng BM25 Index...")
    build_bm25_index(
        map_path=config.MAP_PATH,
        bm25_index_path=config.BM25_INDEX_PATH
    )
    print("Hoàn tất Bước 4.")
    print("\n--- QUÁ TRÌNH INDEXING HOÀN TẤT ---")

if __name__ == "__main__":
    main()