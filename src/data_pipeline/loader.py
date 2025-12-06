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
    SUMMARY_PROMPT_CV,
)

# ============================= PDF LOADING ============================= #
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text, or empty string if error occurs.
    """
    all_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                all_text += page.extract_text() or ""
        return all_text
    except Exception as e:
        print(f"(X) Error processing {pdf_path}: {e}")
        return ""
    
# ============================= CV → JSON PROCESSING ============================= #   
def process_cvs_to_json(cv_folder_path: str, json_output_folder: str):
    """
    Process CV PDFs into structured JSON using Gemini.

    Args:
        cv_folder_path (str): Path to folder containing CV PDFs.
        json_output_folder (str): Path to save extracted JSON files.
    """
    from .loader import extract_text_from_pdf

    if not os.path.exists(json_output_folder):
        os.makedirs(json_output_folder)

    files = [f for f in os.listdir(cv_folder_path) if f.endswith(".pdf")]
    print(f"🔍 Found {len(files)} CVs. Starting extraction with Gemini...")

    for filename in tqdm(files, desc="Gemini Extraction"):
        pdf_path = os.path.join(cv_folder_path, filename)
        json_filename = os.path.splitext(filename)[0] + ".json"
        json_path = os.path.join(json_output_folder, json_filename)

        # Skip if JSON already exists
        if os.path.exists(json_path):
            continue

        # 1. Read PDF
        cv_text = extract_text_from_pdf(pdf_path)
        if not cv_text:
            continue

        # 2. Call Gemini for JSON extraction
        json_data = call_gemini_json(PROMPT_CV, cv_text)

        if json_data:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)

        # Avoid hitting API rate limits
        time.sleep(1)

    print("(V) Completed CV → JSON extraction.")

# ============================= JSON → SUMMARY PROCESSING ============================= #
def generate_summaries(json_input_folder: str, summary_output_folder: str):
    """
    Generate summaries from structured CV JSON using Groq.

    Args:
        json_input_folder (str): Path to folder containing CV JSON files.
        summary_output_folder (str): Path to save summary files.
    """
    print("🚀 Starting summary generation with Groq (Llama/Qwen)...")
    os.makedirs(summary_output_folder, exist_ok=True)

    json_files = [f for f in os.listdir(json_input_folder) if f.endswith(".json")]

    for json_filename in tqdm(json_files, desc="Groq Summarization"):
        input_path = os.path.join(json_input_folder, json_filename)
        summary_filename = os.path.splitext(json_filename)[0] + ".summary"
        output_path = os.path.join(summary_output_folder, summary_filename)

        # Skip if summary already exists
        if os.path.exists(output_path):
            continue

        try:
            # 1. Read JSON
            with open(input_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            # 2. Call Groq for summarization
            summary_text = call_groq_summary(SUMMARY_PROMPT_CV, json_data)

            if summary_text:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(summary_text)
                print(f"(V) Summary created for {json_filename}")
            else:
                print(f"(X) Failed {json_filename}: API returned None")

        except Exception as e:
            print(f"(X) Error processing {json_filename}: {e}")

        # Avoid hitting API rate limits
        time.sleep(10)

    print("(V) Completed summary generation.")