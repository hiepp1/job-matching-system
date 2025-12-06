import os
import time
import json
import pdfplumber
import google.generativeai as genai

from tqdm import tqdm
from ..core.llm import (
    call_gemini_json, 
    call_groq_summary, 
    PROMPT_CV, 
    SUMMARY_PROMPT_CV
)

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file and returns the text."""
    all_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                all_text += page.extract_text() or ""
        return all_text
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return ""
    
def process_cvs_to_json(cv_folder_path: str, json_output_folder: str):
    from .loader import extract_text_from_pdf 
    
    if not os.path.exists(json_output_folder):
        os.makedirs(json_output_folder)

    files = [f for f in os.listdir(cv_folder_path) if f.endswith('.pdf')]
    print(f"🔍 Tìm thấy {len(files)} CV. Bắt đầu trích xuất với GEMINI...")

    for filename in tqdm(files, desc="Gemini Extraction"):
        pdf_path = os.path.join(cv_folder_path, filename)
        json_filename = os.path.splitext(filename)[0] + ".json"
        json_path = os.path.join(json_output_folder, json_filename)

        if os.path.exists(json_path): continue 

        # 1. Đọc PDF
        cv_text = extract_text_from_pdf(pdf_path)
        if not cv_text: continue

        # 2. Gọi Gemini trích xuất JSON
        json_data = call_gemini_json(PROMPT_CV, cv_text)
        
        if json_data:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)
        
        time.sleep(1) 

    print("✅ Hoàn tất trích xuất JSON.")

def generate_summaries(json_input_folder: str, summary_output_folder: str):
    print(f"🚀 Bắt đầu tạo summary với GROQ (Llama/Qwen)...")
    os.makedirs(summary_output_folder, exist_ok=True)
    
    json_files = [f for f in os.listdir(json_input_folder) if f.endswith('.json')]
    
    for json_filename in tqdm(json_files, desc="Groq Summarization"):
        input_path = os.path.join(json_input_folder, json_filename)
        summary_filename = os.path.splitext(json_filename)[0] + ".summary"
        output_path = os.path.join(summary_output_folder, summary_filename)

        if os.path.exists(output_path): continue

        try:
            # 1. Đọc JSON
            with open(input_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            # 2. Gọi Groq tóm tắt
            summary_text = call_groq_summary(SUMMARY_PROMPT_CV, json_data)
            
            if summary_text:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(summary_text)
                print(f"✅ Đã tạo summary cho {json_filename}")
            else:
                print(f"❌ Thất bại file {json_filename}: API trả về None (Xem lỗi chi tiết ở trên)")               
 
        except Exception as e:
            print(f"❌ Lỗi file {json_filename}: {str(e)}")
        
        time.sleep(10) 

    print("✅ Hoàn tất tạo summary.")