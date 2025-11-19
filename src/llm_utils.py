import os
import json
import faiss
import time

import google.generativeai as genai
from groq import Groq

from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

from .indexing import GLOBAL_MODEL

load_dotenv()

# Config Gemini (for extract JSON)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_2")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY1 not found in .env file")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# Config Groq (for summarization
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"

## ============================= CV PROCESSING ========================== ##

## Sửa lại PROMPT_CV: Yêu cầu AI quy đổi mọi loại chứng chỉ (hoặc mô tả "thành thạo") 
# về Thang đo CEFR (A1, A2, B1, B2, C1, C2) -- FIXED
PROMPT_CV = """
You are an Expert IT Technical Recruiter and CV Parser.
Your task is to extract structured data from a candidate's CV text.
The output must be highly granular and specific to the Information Technology domain.

**INPUT TEXT:**
{cv_text}

**STRICT EXTRACTION RULES:**
1. **No Hallucination:** If information (dates, GPA, scores) is missing, return `null` or empty strings. DO NOT guess.
2. **Language:** Output must be in **English**.
3. **Tech Stack Categorization:** You must classify skills into strict categories (Languages, Frameworks, Tools, Cloud/DevOps, Concepts).
4. **Experience Separation:** Strictly distinguish between "Commercial Work" (Employment) and "Projects" (Academic/Personal).
5. **Normalization (CRITICAL):**
   - **Location:** Normalize ANY location to the official English Province/City name (e.g., "Saigon", "Q1" -> "Ho Chi Minh City, Vietnam"; "Hanoi" -> "Hanoi, Vietnam").
   - **English:** Convert ALL English indicators (IELTS, TOEIC, "Fluent", "Basic") into a **CEFR Level** (A1, A2, B1, B2, C1, C2).
     - Example: IELTS 6.0 -> B2; TOEIC 800 -> C1; "Fluent" -> C1; "Basic" -> A2.

**REQUIRED JSON STRUCTURE (Example):**
```json
{
  "candidate_profile": {
    "name": "Full Name",
    "current_location": "Standardized City Name (e.g., "Ho Chi Minh City, Vietnam", "Da Nang, Vietnam")",
    "role_focus": "Main role inferred (e.g., Frontend Dev, AI Engineer, DevOps)",
    "seniority_level": "Inferred level (Intern/Fresher/Junior/Middle/Senior/Lead)"
  },
  "metrics": {
    "education_level": "Highest degree (e.g., Bachelor, Master, PhD)",
    "english_proficiency": "Inferred (e.g., Basic, Intermediate, Advanced/IELTS 6.0+)",
    "gpa": "GPA score if available else null",
    "english_cefr_level": "String (A1/A2/B1/B2/C1/C2/Native or null)",
    "years_experience": 0.0
  },
  "tech_stack": {
    "programming_languages": ["Python", "Java", "C++", "JavaScript"],
    "frameworks_libraries": ["React", "PyTorch", "Spring Boot", "Flutter"],
    "databases": ["MySQL", "MongoDB", "PostgreSQL"],
    "devops_cloud": ["AWS", "Docker", "Kubernetes", "Firebase", "CI/CD"],
    "tools_platforms": ["Git", "Jira", "Figma", "VS Code"],
    "architectures_models": [
            // List Architectures (CNN, RNN, LSTM, Transformer, Microservices)
            // List Specific Models (BERT, GPT-4, LLaMA, ResNet, YOLO)
            "CNN", "Transformer", "ResNet"
        ],
    "techniques_concepts": [
        // List Methodologies & Techniques
        "RAG", "Fine-tuning", "Prompt Engineering", "Computer Vision", "NLP", 
        "Agile", "Scrum", "OOP", "RESTful"
    ]
  },
  "work_experience": [
    {
      "company": "Company Name",
      "role": "Job Title",
      "start_date": "YYYY-MM or null",
      "end_date": "YYYY-MM or 'Present' or null",
      "duration_months": 0,
      "is_internship": false,
      "tech_used": ["List", "of", "tech", "used"],
      "responsibilities": "Brief summary of tasks"
    }
  ],
  "projects": [
    {
      "name": "Project Name",
      "type": "Academic/Personal/Hackathon",
      "tech_used": ["List", "of", "tech", "used"],
      "description": "Short description of what was built"
    }
  ],
  "education": [
    {
      "institution": "University Name",
      "degree": "Degree Name",
      "major": "Major",
      "graduation_year": "YYYY or null",
      "status": "Graduated/In Progress"
    }
  ],
  "certifications": [
    {
      "name": "Cert Name",
      "issuer": "Organization",
      "year": "YYYY or null"
    }
  ]
}
"""

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
You are an Expert Technical Analyst. Your task is to generate a **dense, keyword-rich technical profile** for an IT candidate based on their structured JSON data.
This summary will be used for **Semantic Search** matching against Job Descriptions.

**INPUT DATA (JSON):**
CV_JSON_DATA

**GENERATION RULES:**
1.  **Role & Seniority First:** Start immediately with the inferred Role Focus and Seniority Level (e.g., "A Junior AI Engineer specialized in...").
2.  **Deep Tech Focus:** Explicitly mention specific **Architectures** (CNN, RNN, Transformer), **Techniques** (RAG, Fine-tuning, CI/CD), and **Core Stacks** (Languages/Frameworks). Do not just list "Machine Learning" generically.
3.  **Contextual Experience:** Distinguish between "Commercial Experience" (Real-world projects) and "Academic/Personal Projects". Highlight key achievements using numbers if available.
4.  **Format:** Write in **2 dense paragraphs** (No bullet points).
    * *Paragraph 1:* Professional Identity, Experience Summary, and Core Expertise (Architectures/Techniques).
    * *Paragraph 2:* Technical Stack details (Languages, Databases, Cloud) and notable Domain Knowledge.

**OUTPUT TEMPLATE:**
"[Name] is a [Seniority] [Role Focus] with [Number] years of commercial experience (if any). Their core expertise lies in [Key Architectures/Concepts] using [Key Frameworks]... (Details on Experience)...

Technically, they are proficient in [Languages] and have hands-on experience with [Cloud/DevOps/Tools]... (Details on Stack)..."
"""

## ============================= JD PROCESSING ========================== ##
PROMPT_JD = """
You are an Expert IT Technical Recruiter. Your task is to extract structured requirements from a Job Description (JD) text.

**INPUT JD TEXT:**
{jd_text}

**EXTRACTION RULES:**
1. **NO SOFT SKILLS:** Do NOT extract skills like "Communication", "Teamwork", "Leadership", "Problem Solving". Ignore them completely.
2. **Granularity:** Extract specific technical keywords (e.g., "Python", "React", "AWS", "Scrum") rather than generic terms.
3. **Mandatory vs Preferred:** Carefully distinguish between "Required/Must-have" skills and "Preferred/Nice-to-have/Plus" skills.
4. **Inference:** - Infer `role_focus` (e.g., Backend, AI, Mobile) based on the tech stack.
   - Infer `seniority_level` (Intern, Fresher, Junior, Middle, Senior, Lead) based on title and experience years.
   - Infer `domain` (e.g., E-commerce, Fintech, Outsourcing) based on company description.
5. **English Level:** Infer the required CEFR Level (A1-C2). If JD says "Read documents" -> B1. If "Fluent communication" -> C1.
6. **Location:** Standardized City/Province/Country name.
7. **Output Language:** English.

**REQUIRED JSON STRUCTURE:**
```json
{
  "job_info": {
    "job_title": "String (e.g., Senior Python Developer)",
    "location": "Standardized City/Country Name (e.g., Ho Chi Minh City, Vietnam)",
    "min_english_cefr": "String (A1/A2/B1/B2/C1/C2/Native or null)",
    "role_focus": "String (e.g., Backend, AI, DevOps)",
    "seniority_level": "String (Intern/Fresher/Junior/Middle/Senior/Lead)",
    "min_years_experience": Float, // e.g., 2.0. Return 0 if not specified.
    "domain": "String (e.g., Fintech, Healthcare, Unknown)"
  },
  "qualifications": [
    // List degree requirements or educational background
    "Bachelor's degree in Computer Science",
    "Master's is a plus"
  ],
  "skills": {
    "required": [
      // CORE Tech Stack (Languages, Frameworks, Databases, Architectures)
      // MUST be strictly technical (e.g., "Java", "Microservices", "Spring Boot")
    ],
    "preferred": [
      // "Nice to have" skills, "Plus", "Advantage"
      // MUST be strictly technical (e.g., "AWS", "Docker")
    ]
  },
  "languages": [
    // Spoken languages required
    "English (IELTS 6.0+)",
    "Vietnamese"
  ],
  "certifications": [
    // Specific certs required or preferred
    "AWS Certified Solutions Architect",
    "PMP"
  ]
}
"""

SUMMARY_PROMPT_JD = """
You are a Technical Hiring Manager. Your task is to generate a **dense, keyword-focused technical summary** of a Job Description based on its structured JSON data.
This summary will be used to match against Candidate CVs using Semantic Search.

**INPUT DATA (JSON):**
{jd_json}

**GENERATION RULES:**
1.  **Context:** Start with the [Seniority] [Role Focus] position in [Domain] domain. Mention [Min Years] years of experience if specified.
2.  **Mandatory Stack (Critical):** Explicitly list the "required" skills from the JSON. Use strong phrasing like "Requires strong proficiency in...", "Must have experience with...".
3.  **Preferred Stack (Bonus):** Mention the "preferred" skills separately. Use phrasing like "Experience with X is a plus", "Knowledge of Y is preferred".
4.  **Qualifications:** Briefly mention degree/certification requirements if they exist.
5.  **Style:** Dense paragraph, no bullet points. Focus strictly on technical keywords (Languages, Frameworks, Tools, Architectures).

**OUTPUT TEMPLATE:**
"Seeking a [Seniority] [Role Focus] ([Job Title]) for the [Domain] sector in [Location]. The role requires a minimum of [Number] years of experience.
Core mandatory technical skills include [List Required Skills]...

Ideally, the candidate should also possess [List Preferred Skills]... Candidates with [Certifications/Languages] are highly valued."
"""

def call_gemini_json(prompt, text_content):
    """Chuyên dùng Gemini để Extract JSON"""
    full_prompt = prompt.replace("{cv_text}", text_content).replace("{jd_text}", text_content)
    
    try:
        response = model.generate_content(
            full_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.0, # 0.0 để cấu trúc chính xác
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"⚠️ Gemini Extract Error: {e}")
        return None

def call_groq_summary(prompt_template, json_data):
    """
    [UPDATED] Chuyển sang dùng Gemini 1.5 Flash để Tóm tắt
    vì Groq đã hết Quota trong ngày (Daily Limit Reached).
    """
    try:
        json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
        
        full_prompt = prompt_template.replace("CV_JSON_DATA", json_str).replace("{jd_json}", json_str)
        
        response = model.generate_content(
            full_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.2
            )
        )
        
        return response.text.strip()

    except Exception as e:
        print(f"❌ Lỗi Gemini Summary: {e}")
        return None
    
def process_jd_query(pdf_path: str) -> dict:
    try:
        from .data_preprocessing import extract_text_from_pdf
    except ImportError:
        from data_preprocessing import extract_text_from_pdf

    print(f"📄 Processing JD: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    if not text: return {}

    print("🤖 Gemini extracting JD structure...")
    jd_data = call_gemini_json(PROMPT_JD, text)
    
    if not jd_data:
        print("❌ Failed to extract JD JSON.")
        jd_data = {}

    print(f"📝 Groq ({GROQ_MODEL_NAME}) summarizing...")
    jd_summary = call_groq_summary(SUMMARY_PROMPT_JD, jd_data)

    return {
        "pdf_path": pdf_path,
        "extracted": jd_data,
        "summary": jd_summary
    }





