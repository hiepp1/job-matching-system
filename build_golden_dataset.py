# build_golden_dataset.py
import os
import json
import time
import random
import re
import pandas as pd
from tqdm import tqdm
import google.generativeai as genai
from dotenv import load_dotenv

# --- 1. CẤU HÌNH & LOAD API KEYS ---
load_dotenv()

# Load danh sách các Key
API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3")
]
# Lọc bỏ các key None (trong trường hợp file .env thiếu)
API_KEYS = [k for k in API_KEYS if k]

if not API_KEYS:
    raise ValueError("❌ Không tìm thấy API Key nào trong file .env!")

print(f"✅ Đã load thành công {len(API_KEYS)} API Keys để xoay vòng.")

# Cấu hình đường dẫn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CV_JSON_FOLDER = os.path.join(BASE_DIR, 'artifacts', 'json_cv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'artifacts', 'golden_dataset.csv')

# --- 2. CLASS QUẢN LÝ API (Key Rotation) ---
class GeminiManager:
    def __init__(self, keys):
        self.keys = keys
        self.current_key_index = 0
        self.model_name = "gemini-2.5-flash" 

    def _get_model(self):
        """Lấy model với key hiện tại"""
        current_key = self.keys[self.current_key_index]
        genai.configure(api_key=current_key)
        return genai.GenerativeModel(self.model_name)

    def _switch_key(self):
        """Chuyển sang key tiếp theo trong danh sách"""
        self.current_key_index = (self.current_key_index + 1) % len(self.keys)
        # print(f"🔄 Switching to API Key #{self.current_key_index + 1}...")

    def generate_content_safe(self, prompt, max_retries=10):
        """Hàm gọi API an toàn với cơ chế Retry + Rotate Key"""
        for attempt in range(max_retries):
            try:
                model = self._get_model()
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                # Nếu lỗi (Rate Limit 429, Server Error 503...), in nhẹ và đổi key
                # print(f"⚠️ Error with Key #{self.current_key_index + 1}: {e}. Retrying...")
                self._switch_key()
                time.sleep(2) # Nghỉ 2s để thở
        
        print("❌ Thất bại sau nhiều lần thử.")
        return None

# Khởi tạo Manager
gemini_manager = GeminiManager(API_KEYS)

# --- 3. CÁC HÀM LOGIC ---

def load_local_cv_data():
    """Đọc CV từ ổ cứng"""
    cv_list = []
    if not os.path.exists(CV_JSON_FOLDER):
        print(f"❌ Không tìm thấy thư mục: {CV_JSON_FOLDER}")
        return []

    files = [f for f in os.listdir(CV_JSON_FOLDER) if f.endswith('.json')]
    print(f"📂 Tìm thấy {len(files)} file CV JSON.")
    
    for filename in tqdm(files, desc="Loading CVs"):
        file_path = os.path.join(CV_JSON_FOLDER, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Tạo text đại diện
                text_rep = f"ID: {filename}\nRole: {', '.join(data.get('normalized_job_titles', []))}\n" \
                           f"Skills: {', '.join(data.get('skills', []))}\n" \
                           f"Exp: {data.get('years_of_experience', 0)} years."
                
                cv_list.append({
                    "id": filename,
                    "content": text_rep,
                    "full_json": data
                })
        except Exception:
            pass
    return cv_list

def generate_synthetic_jds(cv_list, num_jds=20):
    """Dùng Gemini để bịa ra JD dựa trên CV thật"""
    synthetic_jds = []
    selected_cvs = random.sample(cv_list, min(num_jds, len(cv_list)))
    
    print(f"\n🤖 Đang sinh {len(selected_cvs)} JD giả lập...")
    
    for cv in tqdm(selected_cvs, desc="Generating JDs"):
        prompt = f"""
        Act as an expert HR Manager. Based on this candidate profile, write a short, realistic Job Description (JD) that this candidate would be a Perfect Match for.
        
        Candidate Profile:
        {json.dumps(cv['full_json'])}
        
        Output Requirement:
        - Output ONLY the JD text.
        - Length: 50-100 words.
        - No intro/outro.
        """
        
        jd_text = gemini_manager.generate_content_safe(prompt)
        if jd_text:
            synthetic_jds.append(jd_text.strip())
            
    return synthetic_jds

def score_batch(jd_text, cv_list):
    """
    Kỹ thuật Batching: Gửi 1 JD + 10 CV cùng lúc.
    Prompt V2: Thêm luật Seniority & Tech Stack chặt chẽ.
    """
    # Chuẩn bị text input
    candidates_text = ""
    for cv in cv_list:
        candidates_text += f"--- START CANDIDATE: {cv['id']} ---\n{cv['content']}\n--- END CANDIDATE ---\n\n"
    
    prompt = f"""
    You are a Strict AI Recruitment Manager. Your job is to evaluate candidates for a specific Job Description (JD).
    
    JOB DESCRIPTION:
    {jd_text}
    
    CANDIDATE LIST:
    {candidates_text}
    
    TASK:
    Rate the relevance of each candidate on a scale from 0.0 to 1.0 based on the following STRICT RULES.
    
    SCORING RULES (MUST FOLLOW):
    1. TECH STACK (50%): 
       - Must match core technologies (e.g., Java, Python, React). 
       - If core skills are missing -> Max Score 0.3.
       
    2. EXPERIENCE & SENIORITY (30% - CRITICAL):
       - JD asks for "Senior"/"Lead" (3+ years) AND Candidate has < 2 years exp -> PENALTY: Max Score 0.4 (Even if skills are perfect).
       - JD asks for "Junior"/"Intern" AND Candidate has > 5 years exp -> PENALTY: Max Score 0.6 (Overqualified).
       - JD asks for "Intern" AND Candidate has 0 years -> FULL SCORE possible.
       
    3. DOMAIN FIT (20%):
       - Example: JD needs "Web" but CV is "Embedded" -> Low score.

    OUTPUT FORMAT (Strict JSON):
    Return a single JSON object where keys are the Candidate IDs and values are the scores (float).
    Example: {{"CV_1.json": 0.8, "CV_2.json": 0.1}}
    DO NOT output Markdown. DO NOT output explanation. ONLY JSON.
    """
    
    response_text = gemini_manager.generate_content_safe(prompt)
    
    if response_text:
        try:
            # Làm sạch response để lấy JSON chuẩn
            json_str = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_str:
                return json.loads(json_str.group())
        except Exception as e:
            print(f"  ⚠️ JSON Parse Error: {e}")
    return {}

# --- 4. HÀM MAIN ---
def main():
    # 1. Load Data
    cv_dataset = load_local_cv_data()
    if not cv_dataset: return

    # 2. Sinh JD giả lập (Khoảng 20 cái để có tập dữ liệu 100-200 dòng)
    NUM_JDS = 20
    synthetic_jds = generate_synthetic_jds(cv_dataset, num_jds=NUM_JDS)
    
    # 3. Chấm điểm (Batching)
    golden_data = []
    BATCH_SIZE = 5 # Gửi 5 CV mỗi lần để Gemini không bị loạn
    
    print(f"\n⚖️ Đang chấm điểm (Mode: Batching + 3-Key Rotation)...")
    
    for jd in tqdm(synthetic_jds, desc="Scoring Batches"):
        # Với mỗi JD, chọn ngẫu nhiên 10 CV để chấm
        # (Đảm bảo có cả match và không match)
        sample_cvs = random.sample(cv_dataset, min(10, len(cv_dataset)))
        
        # Chia nhỏ thành các batch nhỏ hơn (ví dụ 5) để gửi
        for i in range(0, len(sample_cvs), BATCH_SIZE):
            batch = sample_cvs[i:i + BATCH_SIZE]
            
            # Gọi Gemini chấm cả batch
            scores_dict = score_batch(jd, batch)
            
            # Lưu kết quả
            for cv in batch:
                # Lấy điểm, nếu lỗi thì mặc định 0.0
                score = scores_dict.get(cv['id'], 0.0)
                
                golden_data.append({
                    "jd_text": jd,
                    "cv_text": cv['content'],
                    "score": score
                })
                
    # 4. Lưu CSV
    if golden_data:
        df = pd.DataFrame(golden_data)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ THÀNH CÔNG! Đã lưu {len(df)} dòng dữ liệu vào:")
        print(f"   -> {OUTPUT_FILE}")
        print("\n5 dòng đầu tiên:")
        print(df.head())
    else:
        print("❌ Không tạo được dữ liệu nào.")

if __name__ == "__main__":
    main()