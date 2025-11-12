import os
import json
import faiss
import google.generativeai as genai
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

from .indexing import GLOBAL_MODEL

# --- TẢI API KEY ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_1")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")
genai.configure(api_key=GEMINI_API_KEY)

# --- KHỞI TẠO MODEL ---
model = genai.GenerativeModel("gemini-2.5-flash")

## ============================= CV PROCESSING ========================== ##
prompt = ''' Extract the information from the given text extracted from a candidate CV and return a JSON object:
{'name':'','email':'','phone':'','skills':'','education':'','experience':'','certifications':'','languages':'',
 'detected_industry':'', 'years_of_experience': 0, 'normalized_job_titles':[]}

Extraction rules:
1. The input text (PDF_TEXT) may be in English or Vietnamese.
2. The entire output, including the contents of all fields (name, descriptions, skills, etc.), **must be strictly in English**. If the source text is in Vietnamese, translate the content for the corresponding fields into English.
3. name – full name of the candidate
4. skills – a list of technical and professional skills, **only list the key skills if there are many skills**, **avoid listing duplicate skills**, **maximum skills listed is 15**
5. education – including degree, institution name, and graduation year
6. experience – for each job: job title, company name, years worked, and a short and concise description
7. certifications – list of certifications, if available
8. languages – languages the candidate can speak or write
9. detected_industry – Choose the *single best* industry for this CV. (Options: 'information_technology', 'finance', 'healthcare', 'education', 'marketing', 'sales', 'logistics_supply_chain', 'human_resources', 'unknown')
10. years_of_experience – Calculate the total years of professional experience as an integer.
11. normalized_job_titles – List the key, standardized job titles from their experience section (e.g., "Software Engineer", "Project Manager").

Mandatory requirements:
1. Ensure each record contains all fields above.
2. If a field is missing, return "N/A", 0, or [] as appropriate.
3. Return a **valid JSON format only**, with no additional descriptions outside the JSON.

Below is the given text extracted:
PDF_TEXT
'''

class EducationItem(BaseModel):
    degree: str
    institution_name: str
    graduation_year: str

class ExperienceItem(BaseModel):
    job_title: str
    company_name: str
    years_worked: str
    description: str

class CandidateInfo(BaseModel):
    name: str
    skills: List[str]
    education: List[EducationItem]
    experience: List[ExperienceItem]
    certifications: List[str]
    languages: List[str]
    detected_industry: str
    years_of_experience: int
    normalized_job_titles: List[str]

SUMMARY_PROMPT_CV = """
You are an experienced Recruitment Specialist. Your task is to review the candidate's CV data (provided in JSON format) and generate a **concise, professional, and compelling summary** based *only* on the provided data.

Summary Requirements:
1.  **Language:** Use standard professional English.
2.  **Length:** The final summary must be between **100 and 150 words**.
3.  **Structure:** The summary must be structured into three distinct sections using bold headings, focusing only on the data provided in the JSON.

**Input CV Data (JSON):**
CV_JSON_DATA

---
**Generate the Summary now:**

**Professional Profile**
[Generate a 3-5 sentence overview detailing the candidate's career. Use the 'years_of_experience' field and the 'experience' array to highlight their primary job titles, total years of experience, and core responsibilities.]

**Core Skills & Technology**
[List 8-10 of the most relevant keywords from the 'skills' array. If the 'skills' array is empty, briefly mention key technologies or methodologies from their 'experience' descriptions.]

**Education & Certifications**
[State the highest or most relevant degree and institution from the 'education' array. List the most important certifications from the 'certifications' array, if available.]
"""

## ============================= JD PROCESSING ========================== ##

prompt_for_jd = """Extract structured information from the following Job Description (JD) text and return a valid JSON in this format:
{
  "job": "",
  "skills": [],
  "detected_industry": "",
  "location": "",
  "experience": "",
  "qualifications": [],
  "language": ""
}

Rules:
1.  **job**, **skills**, and **detected_industry** are **MANDATORY**.
2.  **detected_industry**: This is **MANDATORY**. Choose the *single best* industry from this list: ['information_technology', 'finance', 'healthcare', 'education', 'marketing', 'sales', 'logistics_supply_chain', 'human_resources', 'unknown'].
3.  Output ""must be in English"".
4.  If a non-mandatory field (location, experience, etc.) is missing, return "N/A" or an empty list [].
5.  The JSON must be valid and clean — no extra commentary.

Below is the extracted text:
PDF_TEXT
"""

SUMMARY_PROMPT_JD = """
You are an AI assistant that creates professional summaries of Job Descriptions for the purpose of AI matching with candidate resumes.

Please read the following structured JD data and generate a concise English summary focusing on:

- Job title and detected_industry
- Required skills, technologies and qualifications
- Experience level
- Any domain/industry context

The summary MUST be written in 1 short paragraph (5-7 sentences), optimized for semantic embedding (clear, factual, no bullet points, no personal tone).

Input JSON:
{jd_json}
"""

def summarize_jd(jd_data: dict) -> str:
    prompt = SUMMARY_PROMPT_JD.replace("{jd_json}", json.dumps(jd_data, ensure_ascii=False, indent=2))
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.2
        )
    )
    return response.text.strip()

def process_jd_query(pdf_path: str) -> dict:
    from .data_preprocessing import extract_text_from_pdf

    print(f"Processing JD file: {pdf_path}")

    # Step 1: Extract text
    text = extract_text_from_pdf(pdf_path)
    if not text:
        raise ValueError("Empty text extracted from JD.")
    print(f"Extracted text length: {len(text)}")

    jd_data = {
        "job": "N/A",
        "location": "N/A",
        "experience": "N/A",
        "skills": [],
        "qualifications": [],
        "language": "N/A",
        "detected_industry": "unknown",
    }

    # Step 2: Structured extraction
    print("\nGemini extraction...")
    prompt = prompt_for_jd.replace("PDF_TEXT", text)
    try:
        resp = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                response_mime_type="application/json"
            )
        )

        print(f"--- DEBUG: RAW API RESPONSE: {resp.text}")
        jd_data = json.loads(resp.text)

    except Exception as e:
        print(f"Extraction or JSON parsing failed: {e}. Using fallback data.")

    # Step 3: Generate summary
    print("\nSummarizing JD content...")
    jd_summary = summarize_jd(jd_data)
    print(f"Summary length: {len(jd_summary)}")
    print(jd_summary)

    # Step 4: Embed summary
    print("\nEncoding summary into vector...")
    jd_vector = GLOBAL_MODEL.encode([jd_summary])
    faiss.normalize_L2(jd_vector)
    print("Embedding vector shape:", jd_vector.shape)

    # Step 5: Return results
    return {
        "pdf_path": pdf_path,
        "extracted": jd_data,
        "summary": jd_summary,
        "vector": jd_vector.tolist()
    }