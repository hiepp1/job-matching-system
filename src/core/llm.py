import os
import json
import faiss
import re

from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

import google.generativeai as genai
from groq import Groq

# ============================= ENVIRONMENT ============================= #
load_dotenv()

# Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_1") or os.getenv("GEMINI_API_KEY_2")
if not GEMINI_API_KEY:
    print("(X) Warning: GEMINI_API_KEY not found. Checking KEY_2...")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_2")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# Secondary Gemini API key
GEMINI_API_KEY_2 = os.getenv("GEMINI_API_KEY_3") or os.getenv("GEMINI_API_KEY_4")
model2 = genai.GenerativeModel("gemini-2.5-flash") if GEMINI_API_KEY_2 else model

# Groq configuration (for summarization)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"

# ============================= PROMPTS ============================= #
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
4. **Experience Separation:** Strictly distinguish between "Commercial Work" (MUST be related to IT industry) and "Projects" (Academic/Personal).
5. **Normalization (CRITICAL):**
   - **Location:** Normalize ANY location to the official English Province/City name (e.g., "Saigon", "Q1" -> "Ho Chi Minh City, Vietnam"; "Hanoi" -> "Hanoi, Vietnam").
   - **English:** Convert ALL English indicators (IELTS, TOEIC, TOEFL,...) into a **CEFR Level** (A1, A2, B1, B2, C1, C2).
     - Example: IELTS 6.0 -> B2; TOEIC 800 -> C1; "Fluent" -> Null "Basic" -> Null.
6. **Seniority Inference Logic (CRITICAL):**
   - Calculate `commercial_years_experience` (excluding academic projects).
   - Map to level based on this strict scale:
     - **< 1 year:** "Intern" (if currently studying) or "Fresher".
     - **1 - 3 years:** "Junior/Middle".
     - **4+ years:** "Senior".
   - **Leadership Roles (Lead, Manager, Head, CTO):**
     - ONLY label as "Lead/Manager" if `commercial_years_experience` >= 5 years.
     - If a candidate claims "Head of AI" but has 1 year exp -> Downgrade to "Junior"
7. **Awards Parsing:** 
  - Extract scholarships, competition awards (Hackathons, ICPC), or "Employee of the Year" into `honors_and_awards`. Do NOT put them in `certifications`
8. **Certifications & Honors Filtering (CRITICAL):**
   - Only include certifications and honors/awards that are directly related to IT, Computer Science, Software Engineering, Data, AI/ML, Cloud, Cybersecurity, or other technical domains.
   - Exclude non-IT items (e.g., language courses, soft skills, sports, arts).
   - If unsure, return `null`.

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
  // Courses, Professional Certs (AWS, Coursera, Udemy,...)
    {
      "name": "Cert Name",
      "issuer": "Organization",
      "year": "YYYY or null"
    }
  ],
  "honors_and_awards": [
    // Competitions, Scholarships, Recognition
    {
      "title": "Award Title",
      "issuer": "Organization",
}
"""

SUMMARY_PROMPT_CV = """
You are an Expert Technical Analyst. Your task is to generate a **fact-based, objective technical profile** for an IT candidate based on their structured JSON data.
This summary will be used for **Semantic Search** matching against Job Descriptions.

**INPUT DATA (JSON):**
CV_JSON_DATA

**GENERATION STRUCTURE:**
1.  **Role & Seniority First:** Start immediately with the inferred Role Focus and Seniority Level (e.g., "A Junior AI Engineer specialized in...").
2.  **Deep Tech Focus:** Explicitly mention specific **Architectures** (CNN, RNN, Transformer), **Techniques** (RAG, Fine-tuning, CI/CD), and **Core Stacks** (Languages/Frameworks). Do not just list "Machine Learning" generically.
3.  **Contextual Experience:** Distinguish between "Commercial Experience" (Real-world projects) and "Academic/Personal Projects". Highlight key achievements using numbers if available.
4.  **Education & Language:** 
    - **Language:** Mention their Language level (e.g., B1, C1) if available.  
    - **Education:** If graduation year equals the current year or previous year, state as "completed". If the CV shows a range (e.g., 2022–present or 2022–(over than current year)), state as "Currently pursuing [Degree/Major]".  
5.  **Format:** Write in **2 dense paragraphs** (No bullet points).
    * *Paragraph 1:* Professional Identity, Experience Summary, and Core Expertise (Architectures/Techniques).
    * *Paragraph 2:* Technical Stack details (Languages, Databases, Cloud) and notable Domain Knowledge.

**STRICT TONE CALIBRATION (CRITICAL):**
Determine the tone based on `years_experience` and `seniority_level`:

1.  **IF Seniority is Intern/Fresher OR Years < 2:**
    * **Tone:** Humble, potential-oriented, academic.
    * **Keywords allowed:** "Familiar with", "Exposed to", "Basic understanding", "Academic knowledge", "Learned", "Assisted in".
    * **FORBIDDEN:** "Expert", "Proficient", "Specialized", "Track record", "Architected".
    * **Project Context:** Must describe projects as "academic assignments" or "personal projects" unless stated otherwise.

2.  **IF Seniority is Junior/Middle (2-5 years):**
    * **Tone:** Competent, execution-oriented.
    * **Keywords allowed:** "Proficient in", "Hands-on experience", "Implemented", "Developed", "Solid grasp of".

3.  **IF Seniority is Senior/Lead (5+ years):**
    * **Tone:** Authoritative, strategic, expert.
    * **Keywords allowed:** "Expertise in", "Architected", "Spearheaded", "Mastery of", "Deep knowledge", "Optimized", "Led".

**OUTPUT STRUCTURE (Choose the template matching the seniority):**

**[OPTION A: FOR INTERN/FRESHER]**
"[Name] is an [Intern/Fresher] [Role Focus] with [Number] years of experience. Their background consists primarily of academic coursework and personal projects related to [Key Concepts]. They hold a [Degree] in [Major], possess [Language Level if any] and get a [Honors and Awards if any].

Technically, they have gained basic exposure to [Languages] and are familiar with tools such as [Tools]. They are developing skills in [Frameworks] through [Project Type]..."

**[OPTION B: FOR EXPERIENCED (Junior/Senior)]**
"[Name] is a [Seniority] [Role Focus] with [Number] years of experience. They have a proven track record in [Key Domain], specializing in [Key Architectures/Techniques]. They hold a [Degree] in [Major] and [Language Level if any].

Technically, they demonstrate strong proficiency in [Languages] and [Frameworks]. They have successfully delivered projects using [Cloud/DevOps] and possess deep expertise in [Database/System Design]..."

**INSTRUCTION:**
Analyze the input JSON. Pick the correct tone and template (only A or B). Generate the summary.
"""

PROMPT_JD = """
You are an Expert IT Technical Recruiter. Your task is to extract structured requirements from a Job Description (JD) text.

**INPUT JD TEXT:**
{jd_text}

**EXTRACTION RULES:**
1. **NO SOFT SKILLS:** Do NOT extract skills like "Communication", "Teamwork", "Leadership", "Problem Solving". Ignore them completely.
2. **NO LOGISTICS:** Do NOT extract availability (e.g., "3-4 days/week"), salary, benefits, or start date. Focus ONLY on capability requirements.
3. **Granularity:** Extract specific technical keywords (e.g., "Python", "React", "AWS", "Scrum") rather than generic terms.
4. **Mandatory vs Preferred:** Carefully distinguish between "Required/Must-have" skills and "Preferred/Nice-to-have/Plus" skills.
5. **Inference:** - Infer `role_focus` (e.g., Backend, AI, Mobile) based on the tech stack.
   - Infer `seniority_level` (Intern, Fresher, Junior, Middle, Senior, Lead) based on title and experience years.
   - Infer `domain` (e.g., E-commerce, Fintech, Outsourcing) based on company description.
6. **English Level:** Infer the required CEFR Level (A1-C2). If JD says "Fluent" or "Read Documents" -> null. If "IELTS 6.0" -> B2.
7. **Location:** Standardized City/Province/Country name.
8. **Output Language:** English.

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
  ],
  "requirements_checklist": [
    // This list is for the "Detailed Analysis" UI (Human Readable)
    {
      "category": "Education",
      "content": "Bachelor's degree in Computer Science, IT, or related field."
    },
    {
      "category": "Experience",
      "content": "Minimum 3 years of experience in Backend Development."
    },
    {
      "category": "Tech Stack",
      "content": "Strong proficiency in Python and Django framework."
    },
    {
      "category": "Language",
      "content": "Good reading comprehension of technical documents (English)."
    }
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

PROMPT_VERIFY_MATCH = """
You are an expert Tech Recruiter. Compare a Candidate Profile against the Job Requirements.

**JOB REQUIREMENTS:**
{jd_requirements}

**CANDIDATE PROFILE:**
{cv_profile}

**INSTRUCTION:**
For each requirement in the list, check if the candidate matches.
**CRITICAL LOGIC:** If a requirement says "A or B or C" (e.g., "Degree in CS, IT, or related"), and the candidate has ANY of them, mark it as **"match"**.

**OUTPUT FORMAT (JSON Only):**
{
  "checks": [
      {
        "category": "Education",
        "requirement": "Bachelor's degree in CS, IT, or related",
        "status": "match", 
        "reason": "Candidate has BS in Information Technology (Matches 'related')."
      }
  ]
}
**STATUS OPTIONS:** "match", "partial", "missing", "unknown"
"""

# ============================= UTILITIES ============================= #
def clean_response_text(text: str) -> str:
    """
    Clean response text from LLM by removing JSON wrappers and unnecessary phrases.
    """
    if not text:
        return ""

    # Remove JSON markdown wrappers
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*$', '', text)

    # Remove redundant leading phrases
    garbage_phrases = [
        "Based on the provided JSON", "Here is a dense", "Here is the summary",
        "Here represents", "The following is", "technical profile"
    ]
    lines = text.split('\n')
    cleaned = [line for line in lines if not any(p.lower() in line.lower() for p in garbage_phrases)]
    return "\n".join(cleaned).strip()

def call_smart_json(prompt: str, text_content: str):
    """
    Call Gemini model to extract structured JSON from CV or JD text.
    """
    full_prompt = prompt.replace("{cv_text}", text_content).replace("{jd_text}", text_content)
    try:
        response = model.generate_content(
            full_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"(X) Gemini Extract Error: {e}")
        return None

def call_smart_summary(prompt_template: str, json_data: dict) -> str:
    """
    Generate a summary using Groq (preferred) or Gemini as fallback.
    """
    json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
    full_prompt = prompt_template.replace("CV_JSON_DATA", json_str).replace("{jd_json}", json_str)

    try:
        response = model.generate_content(
            full_prompt,
            generation_config=genai.GenerationConfig(temperature=0.3)
        )
        return clean_response_text(response.text)
    except Exception as e:
        print(f"(X) All AI Models Failed (Summary): {e}")
        return ""
    
def call_smart_verification(jd_text: str, cv_text: str) -> list:
    """
    Compare JD requirements against CV profile and return structured verification checks.
    """
    full_prompt = PROMPT_VERIFY_MATCH.replace("{jd_requirements}", jd_text).replace("{cv_profile}", cv_text)

    # Prefer Groq for JSON mode
    if groq_client:
        try:
            completion = groq_client.chat.completions.create(
                model=GROQ_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an objective evaluator. Output JSON only."},
                    {"role": "user", "content": full_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            data = json.loads(completion.choices[0].message.content)
            return data.get("checks", [])
        except Exception as e:
            print(f"(X) Groq Verify Error: {e}")

    # Fallback Gemini
    try:
        response = model.generate_content(
            full_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                response_mime_type="application/json"
            )
        )
        data = json.loads(response.text)
        return data.get("checks", []) if isinstance(data, dict) else data
    except Exception as e:
        print(f"(X) Gemini Verify Error: {e}")
        return []

call_gemini_json = call_smart_json
call_groq_summary = call_smart_summary

# ============================= JD PROCESSING ============================= #
def process_jd_query(pdf_path: str) -> dict:
    """
    Process a Job Description PDF:
    - Extract text
    - Generate structured JD JSON
    - Produce JD summary
    """
    try:
        from ..data_pipeline.loader import extract_text_from_pdf
    except ImportError:
        from data_pipeline.loader import extract_text_from_pdf

    print(f"Processing JD: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return {"extracted": {}, "summary": ""}

    print("Extracting JD structure...")
    jd_data = call_gemini_json(PROMPT_JD, text) or {}

    print("Generating JD summary...")
    jd_summary = call_groq_summary(SUMMARY_PROMPT_JD, jd_data)

    return {
        "pdf_path": pdf_path,
        "extracted": jd_data,
        "summary": jd_summary
    }