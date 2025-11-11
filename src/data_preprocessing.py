import os
import time
import json
import pdfplumber
from tqdm import tqdm
import google.generativeai as genai

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
    from .llm_utils import model, prompt, CandidateInfo
    """
    Đọc tất cả file PDF từ cv_folder_path, trích xuất thông tin
    bằng Gemini và lưu file .json vào json_output_folder.
    """
    
    print(f"Bắt đầu xử lý CVs từ: {cv_folder_path}")
    files = sorted(os.listdir(cv_folder_path))
    files_path = [os.path.join(cv_folder_path, file_name) for file_name in files if file_name.endswith('.pdf')]
    print(f"Tìm thấy {len(files_path)} file PDF.")

    extracted_dict = {
        os.path.basename(file): extract_text_from_pdf(file)
        for file in tqdm(files_path, desc="Đang trích xuất text từ PDF")
    }

    os.makedirs(json_output_folder, exist_ok=True)
    
    all_results = {}
    
    for original_filename, pdf_text in tqdm(extracted_dict.items(), desc="Đang trích xuất JSON bằng Gemini"):
        
        if not pdf_text:
            print(f"Skipping {original_filename}: No text.")
            all_results[original_filename] = "EXTRACTION_FAILED_PDF"
            continue

        base_name, _ = os.path.splitext(original_filename)
        json_filename = base_name + ".json"
        output_filepath = os.path.join(json_output_folder, json_filename)

        if os.path.exists(output_filepath):
            all_results[original_filename] = "SKIPPED_EXISTS"
            continue

        full_prompt = prompt.replace('PDF_TEXT', pdf_text)
        print(f"\nProcessing new file: {original_filename}")
        try:
            result = model.generate_content(
                full_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.7,
                    response_mime_type="application/json",
                    response_schema=CandidateInfo,
                ),
            )
            all_results[original_filename] = result.text
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(result.text)
            print(f"Successfully saved output for {original_filename}")
        except Exception as e:
            print(f"An error occurred for {original_filename}: {e}")
            all_results[original_filename] = f"API_ERROR: {str(e)}"
        
        time.sleep(5) # Giữ delay
    
    print("Hoàn tất trích xuất JSON.")
    return all_results

def generate_summaries(json_input_folder: str, summary_output_folder: str):
    from .llm_utils import model, SUMMARY_PROMPT_CV
    """
    Đọc tất cả file .json từ json_input_folder, tạo summary
    bằng Gemini và lưu file .summary vào summary_output_folder.
    """
    
    print(f"Bắt đầu tạo summary từ: {json_input_folder}")
    os.makedirs(summary_output_folder, exist_ok=True)
    all_summaries = {}
    delay_time = 5
    
    json_files = [f for f in os.listdir(json_input_folder) if f.endswith('.json')]
    
    for json_filename in tqdm(json_files, desc="Đang tạo summaries"):
        input_filepath_json = os.path.join(json_input_folder, json_filename)
        base_name, _ = os.path.splitext(json_filename)
        summary_filename = base_name + ".summary"
        output_filepath_summary = os.path.join(summary_output_folder, summary_filename)

        if os.path.exists(output_filepath_summary):
            continue

        print(f"\nProcessing new file: {json_filename}")
        try:
            with open(input_filepath_json, 'r', encoding='utf-8') as f:
                json_string = f.read()

            summary_request_prompt = SUMMARY_PROMPT_CV.replace('CV_JSON_DATA', json_string)
            
            summary_result = model.generate_content(
                summary_request_prompt,
                generation_config=genai.GenerationConfig(temperature=0.3),
            )
            summary_text = summary_result.text

            with open(output_filepath_summary, 'w', encoding='utf-8') as f:
                f.write(summary_text)
            print(f"\nSummary saved for {json_filename}")
            all_summaries[json_filename] = summary_text
        except Exception as e:
            error_message = f"ERROR PROCESSING {json_filename}: {str(e)}"
            print(f"❌ {error_message}")
            all_summaries[json_filename] = error_message
        
        time.sleep(delay_time)

    print("Hoàn tất tạo summary.")
    return all_summaries