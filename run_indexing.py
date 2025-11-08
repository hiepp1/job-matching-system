# run_indexing.py
import os
import config # Import file config
from src.data_preprocessing import process_cvs_to_json, generate_summaries
from src.indexing import build_rich_faiss_index_offline, build_bm25_index

def main():
    print("--- BẮT ĐẦU QUÁ TRÌNH INDEXING ---")

    # 1. Đảm bảo các thư mục tồn tại
    os.makedirs(config.CV_JSON_FOLDER, exist_ok=True)
    os.makedirs(config.CV_SUMMARY_FOLDER, exist_ok=True)
    os.makedirs(config.CV_DATASET_FOLDER, exist_ok=True)
    print("Đã kiểm tra/tạo các thư mục artifacts.")

    # 2. Chạy Bước I: PDF -> JSON
    print("\n[Bước 1/4] Đang xử lý CVs PDF sang JSON...")
    process_cvs_to_json(
        cv_folder_path=config.CV_FOLDER,
        json_output_folder=config.CV_JSON_FOLDER
    )
    print("Hoàn tất Bước 1.")

    # 3. Chạy Bước II: JSON -> Summary
    print("\n[Bước 2/4] Đang tạo Summaries từ JSON...")
    generate_summaries(
        json_input_folder=config.CV_JSON_FOLDER,
        summary_output_folder=config.CV_SUMMARY_FOLDER
    )
    print("Hoàn tất Bước 2.")

    # 4. Chạy Bước V: Tạo FAISS Index
    print("\n[Bước 3/4] Đang xây dựng FAISS Index...")
    build_rich_faiss_index_offline(
        summary_folder=config.CV_SUMMARY_FOLDER,
        json_folder=config.CV_JSON_FOLDER,
        pdf_folder=config.CV_FOLDER,
        output_folder=config.CV_DATASET_FOLDER,
        dimension=config.EMBEDDING_DIMENSION
    )
    print("Hoàn tất Bước 3.")

    # 5. Chạy Bước VI: Tạo BM25 Index
    print("\n[Bước 4/4] Đang xây dựng BM25 Index...")
    build_bm25_index(
        map_path=config.MAP_PATH,
        bm25_index_path=config.BM25_INDEX_PATH
    )
    print("Hoàn tất Bước 4.")
    
    print("\n--- QUÁ TRÌNH INDEXING HOÀN TẤT ---")

if __name__ == "__main__":
    main()