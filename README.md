---
title: Job Matching Demo
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

#  AI Job Matching System

Đây là dự án demo cho một hệ thống gợi ý CV thông minh, sử dụng AI tạo sinh (Gemini) và kỹ thuật tìm kiếm lai (Hybrid Search) để tìm kiếm các CV phù hợp nhất với một bản mô tả công việc (JD) được cung cấp.

## ✨ Tính năng chính

* **Trích xuất thông minh:** Sử dụng Google Gemini 1.5 Flash để đọc và trích xuất thông tin có cấu trúc (kỹ năng, kinh nghiệm, học vấn) từ cả CV và JD (dạng PDF).
* **Tìm kiếm lai (Hybrid Search):** Kết hợp sức mạnh của:
    * **Tìm kiếm ngữ nghĩa (Semantic Search):** Dùng `FAISS` và mô hình embedding (`bge-base-en-v1.5`) để hiểu *ý nghĩa* của CV và JD.
    * **Tìm kiếm từ khóa (Keyword Search):** Dùng `BM25` để đảm bảo các *từ khóa* quan trọng (như "Python", "TensorFlow") được ưu tiên.
* **Xếp hạng nâng cao:** Sử dụng `Cross-Encoder` để xếp hạng lại (re-rank) các kết quả hàng đầu, đảm bảo độ chính xác cao nhất.
* **Giao diện trực quan:** Giao diện `Gradio` đơn giản cho phép người dùng tải JD lên và nhận kết quả ngay lập tức.
* **Triển khai MLOps:** Dự án được cấu trúc theo chuẩn MLOps-Lite, với pipeline xử lý dữ liệu (`run_indexing.py`) và pipeline phục vụ (`app.py`) riêng biệt, được triển khai tự động qua `Hugging Face Spaces`.

## ⚙️ Công nghệ sử dụng (Tech Stack)

* **Ngôn ngữ:** Python
* **LLM & Embedding:** Google Gemini API, `sentence-transformers`
* **Vector Database & Search:** `FAISS`, `rank_bm25`
* **Giao diện (UI):** `Gradio`
* **Triển khai (Deployment):** `Hugging Face Spaces`, `Git LFS`

## 🚀 Cách thức hoạt động (Pipeline)

1.  **ETL (Trích xuất):** Các CV (PDF) được đọc và gửi đến Gemini API để trích xuất JSON, sau đó được tóm tắt.
2.  **Indexing (Xây dựng chỉ mục):** Các bản tóm tắt được "vector hóa" và lưu vào `FAISS` (cho tìm kiếm ngữ nghĩa) và `BM25` (cho tìm kiếm từ khóa). Các file index này được lưu trữ bằng `Git LFS`.
3.  **Serving (Phục vụ):** Khi người dùng tải lên một JD:
    * JD được xử lý tương tự (Trích xuất -> Tóm tắt -> Vector hóa).
    * Hệ thống thực hiện tìm kiếm lai trên các index đã được build.
    * Các ứng viên hàng đầu được xếp hạng lại và hiển thị trên giao diện Gradio.

## 📋 Cách sử dụng Demo

**Try the live application here:**
[My Job Matching System Demo](https://huggingface.co/spaces/leviethiep/My-Job-Matching-System)

1.  Tải lên một file Mô tả công việc (Job Description) định dạng `.pdf`.
2.  Nhấn nút "Submit".
3.  Chờ hệ thống xử lý JD và tìm kiếm.
4.  Xem 10 CV phù hợp nhất trong bảng kết quả.