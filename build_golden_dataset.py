import os
import json
import time
import random
import re
import pandas as pd
from tqdm import tqdm
import google.generativeai as genai
from dotenv import load_dotenv

# --- 1. CẤU HÌNH ---
load_dotenv()
# Load 3 Keys
API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"), 
    os.getenv("GEMINI_API_KEY_2"), 
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4")
]
API_KEYS = [k for k in API_KEYS if k]

if not API_KEYS: 
    raise ValueError("❌ Lỗi: Không tìm thấy API Key nào!")

print(f"✅ Đã load {len(API_KEYS)} API Keys. Chế độ: Batch Processing.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CV_JSON_FOLDER = os.path.join(BASE_DIR, 'artifacts', 'json_cv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'artifacts', 'golden_dataset_v2.csv')

# --- 2. CLASS QUẢN LÝ API ---
class GeminiManager:
    def __init__(self, keys):
        self.keys = keys
        self.index = 0
        self.model_name = "gemini-2.5-flash"

    def generate_batch(self, prompt):
        """Hàm gọi API có Retry & Rotation"""
        for _ in range(len(self.keys) * 2): # Thử xoay vòng 2 lượt
            try:
                genai.configure(api_key=self.keys[self.index])
                model = genai.GenerativeModel(self.model_name)
                # Ép trả về JSON
                response = model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(response_mime_type="application/json")
                )
                return json.loads(response.text)
            except Exception as e:
                print(f"  ⚠️ Key #{self.index+1} bị lỗi/limit. Đổi key...")
                self.index = (self.index + 1) % len(self.keys)
                time.sleep(5) # Nghỉ 5s trước khi thử key mới
        return {}

manager = GeminiManager(API_KEYS)

# --- 3. HÀM LOAD DỮ LIỆU ---
def load_cvs(limit=60):
    cvs = []
    if not os.path.exists(CV_JSON_FOLDER): return []
    
    files = [f for f in os.listdir(CV_JSON_FOLDER) if f.endswith('.json')]
    selected_files = random.sample(files, min(limit, len(files)))
    
    print(f"📂 Đang đọc {len(selected_files)} file CV...")
    
    for f in selected_files:
        try:
            with open(os.path.join(CV_JSON_FOLDER, f), 'r', encoding='utf-8') as file:
                data = json.load(file)
                # Lấy dữ liệu tinh gọn
                profile = data.get('candidate_profile', {})
                tech = data.get('tech_stack', {})
                
                langs = tech.get('programming_languages', [])
                libs = tech.get('frameworks_libraries', [])
                
                if not langs and not libs: continue
                
                # Text đại diện
                text_rep = f"Role: {profile.get('role_focus')} ({profile.get('seniority_level')}). Skills: {', '.join(langs + libs)}."
                
                # ID ngắn gọn để làm key cho JSON
                cv_id = f.replace(".json", "")
                cvs.append({"id": cv_id, "json": data, "text": text_rep})
        except: pass
    return cvs

# --- 4. PROMPTS BATCHING (Xử lý nhiều CV 1 lúc) ---

PROMPT_TEMPLATE = """
You are a Technical Recruiter. I will provide a list of Candidate Profiles (ID and Details).
Your task is to write a specific Job Description (JD) for EACH candidate based on the instructions below.

INSTRUCTION TYPE: {type_instruction}

CANDIDATE LIST:
{candidates_text}

OUTPUT REQUIREMENT:
Return a JSON object where keys are "Candidate_ID" and values are the "JD Text".
Example:
{{
  "CV_1": "JD text for CV 1...",
  "CV_2": "JD text for CV 2..."
}}
"""

# Các chỉ dẫn cụ thể (Inject vào {type_instruction})
INSTRUCTIONS = {
    "Positive": """
    - Write a JD that is a **PERFECT 100% MATCH** for the candidate.
    - Use the EXACT Job Title, Seniority, and Tech Stack from the profile.
    - Length: 50 words.
    """,
    
    "Partial": """
    - Write a JD that is a **PARTIAL MATCH (60%)**.
    - Keep the Seniority MATCHING.
    - Require 50% of their skills, but ADD 2-3 NEW skills they DO NOT have (e.g., if Python, ask for Go; if AWS, ask for Azure).
    - Length: 50 words.
    """,
    
    "Seniority_Mismatch": """
    - Write a JD with a **SERIOUS SENIORITY MISMATCH**.
    - If candidate is Intern/Junior -> Write JD for "Senior Lead/Manager" (7+ years exp).
    - If candidate is Senior -> Write JD for "Unpaid Intern".
    - KEEP the Tech Stack MATCHING (to confuse the AI).
    - Length: 50 words.
    """,
    
    "Tech_Mismatch": """
    - Write a JD with a **COMPLETE TECH MISMATCH**.
    - Keep the Seniority Level similar.
    - CHANGE the Domain entirely (e.g., Web -> Embedded, AI -> Mobile).
    - Length: 50 words.
    """
}

SCORES = {
    "Positive": 1.0,
    "Partial": 0.6,
    "Seniority_Mismatch": 0.2,
    "Tech_Mismatch": 0.1
}

# --- 5. MAIN LOOP (BATCH PROCESSING) ---
def main():
    cv_list = load_cvs(limit=50) 
    if not cv_list: return print("❌ Không có dữ liệu CV.")

    dataset = []
    BATCH_SIZE = 5 # Gửi 5 CV một lúc -> Giảm 5 lần số request
    
    # Chia CV thành các batch
    batches = [cv_list[i:i + BATCH_SIZE] for i in range(0, len(cv_list), BATCH_SIZE)]
    
    print(f"🚀 Bắt đầu sinh dữ liệu: {len(cv_list)} CVs -> {len(batches)} Batches.")
    
    for batch in tqdm(batches, desc="Processing Batches"):
        # Chuẩn bị text đầu vào cho cả batch
        batch_text_input = ""
        for cv in batch:
            # Rút gọn JSON để tiết kiệm token
            mini_profile = {
                "role": cv['json'].get('candidate_profile', {}).get('role_focus'),
                "level": cv['json'].get('candidate_profile', {}).get('seniority_level'),
                "skills": cv['text']
            }
            batch_text_input += f"ID: {cv['id']}\nPROFILE: {json.dumps(mini_profile)}\n---\n"

        # Duyệt qua 4 loại kịch bản (Positive, Partial...)
        for label_type, instruction in INSTRUCTIONS.items():
            # Tạo prompt
            full_prompt = PROMPT_TEMPLATE.format(
                type_instruction=instruction,
                candidates_text=batch_text_input
            )
            
            # Gọi API
            results_dict = manager.generate_batch(full_prompt)
            
            # Map kết quả vào dataset
            if results_dict:
                for cv in batch:
                    # Tìm JD tương ứng với ID
                    jd_text = results_dict.get(cv['id'])
                    if jd_text:
                        dataset.append({
                            "jd_text": jd_text,
                            "cv_text": cv['text'],
                            "score": SCORES[label_type],
                            "type": label_type
                        })
            
            # Nghỉ nhẹ giữa các loại kịch bản
            time.sleep(2)

        # Nghỉ lớn sau mỗi batch CV để reset Rate Limit
        # print("⏳ Cooling down (10s)...")
        time.sleep(10)

    # Lưu kết quả
    if dataset:
        df = pd.DataFrame(dataset)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ THÀNH CÔNG! Đã lưu {len(df)} dòng dữ liệu.")
        print(f"   -> {OUTPUT_FILE}")
        print("\nPhân phối dữ liệu:")
        print(df['type'].value_counts())
    else:
        print("❌ Thất bại: Không sinh được dữ liệu nào.")

if __name__ == "__main__":
    main()