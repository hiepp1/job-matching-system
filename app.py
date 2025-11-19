import gradio as gr
import pandas as pd
import time
import os
import json
from dotenv import load_dotenv

# Load API Keys
load_dotenv()

# Import modules
import config
from src.llm_utils import process_jd_query
from src.search import hybrid_search_v2

# --- CẤU HÌNH ---
TOP_N = 15

# --- HELPER: TẠO BẢNG HTML (SOFT PASTEL PINK) ---
def convert_df_to_html(df, drive_folder_url):
    html = """
    <style>
        .pro-table { 
            width: 100%; 
            border-collapse: collapse; 
            font-family: 'Segoe UI', Helvetica, Arial, sans-serif; 
            font-size: 0.95rem; 
            color: #333;
            background-color: #fff;
            border: 1px solid #fce7f3; /* Viền bảng hồng rất nhạt */
        }
        /* Header màu Hồng Phấn Nhạt */
        .pro-table th { 
            text-align: left; 
            padding: 12px 15px; 
            border-bottom: 2px solid #f472b6; /* Pink-400 */
            background-color: #fdf2f8; /* Pink-50 (Rất nhạt) */
            color: #9d174d; /* Pink-800 (Chữ đậm để dễ đọc) */
            font-weight: 700; 
            text-transform: uppercase;
            font-size: 0.85rem;
            letter-spacing: 0.5px;
        }
        .pro-table td { 
            padding: 12px 15px; 
            border-bottom: 1px solid #f3f4f6; 
            vertical-align: top;
        }
        .pro-table tr:last-child td { border-bottom: none; }
        .pro-table tr:hover { background-color: #fff1f2; } 
        
        /* Tên ứng viên */
        .candidate-name { font-weight: 700; font-size: 1rem; color: #1f2937; margin-bottom: 4px; }
        .candidate-role { font-size: 0.85rem; color: #64748b; margin-bottom: 6px; }
        
        /* Link CV - Nút nhỏ gọn */
        .cv-link a { 
            color: #db2777; 
            text-decoration: none; 
            font-size: 0.75rem; 
            font-weight: 600;
            border: 1px solid #fbcfe8; 
            padding: 3px 8px;
            border-radius: 4px;
            background-color: #fff;
            transition: all 0.2s;
        }
        .cv-link a:hover { background-color: #fce7f3; border-color: #db2777; }

        /* Điểm số */
        .score-val { font-weight: 800; font-size: 1.1rem; color: #be185d; }

        /* Skills Tags - Sạch sẽ */
        .skill-tag { 
            display: inline-block; 
            border: 1px solid #e2e8f0; 
            color: #475569; 
            padding: 3px 8px; 
            border-radius: 4px; 
            font-size: 0.8rem; 
            margin: 2px; 
            background-color: #f8fafc;
            font-weight: 500;
        }
        /* Skill trùng khớp - Hồng nhạt */
        .skill-match {
            border-color: #fbcfe8;
            background-color: #fdf2f8;
            color: #be185d;
        }
        .no-skill { color: #94a3b8; font-style: italic; font-size: 0.85rem; }
    </style>
    
    <table class='pro-table'>
    <thead>
        <tr>
            <th style="width: 40%">Candidate Profile</th>
            <th style="width: 15%">Score</th>
            <th style="width: 45%">Matching Skills</th>
        </tr>
    </thead>
    <tbody>
    """
    
    for _, row in df.iterrows():
        common_skills = row['Common Skills']
        if common_skills:
            skills_html = "".join([f"<span class='skill-tag skill-match'>{s}</span>" for s in common_skills.split(', ')])
        else:
            skills_html = "<span class='no-skill'>No direct match</span>"
        
        cv_link_html = f"<span class='cv-link'><a href='{drive_folder_url}' target='_blank'>View CV PDF</a></span>"

        html += f"""
        <tr>
            <td>
                <div class="candidate-name">{row['Candidate Name']}</div>
                <div class="candidate-role">{row['Job']}</div>
                {cv_link_html}
            </td>
            <td><div class="score-val">{row['Match Score']}</div></td>
            <td>{skills_html}</td>
        </tr>
        """
    html += "</tbody></table>"
    return html

# --- METADATA HELPERS ---
def get_metadata_from_json(summary_filename):
    try:
        base_name = os.path.splitext(summary_filename)[0]
        json_path = os.path.join(config.CV_JSON_FOLDER, f"{base_name}.json")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            profile = data.get("candidate_profile", {})
            name = profile.get("name", "Unknown").upper()
            role = profile.get('role_focus', 'N/A')
            level = profile.get('seniority_level', '')
            full_role = f"{role} • {level}" if level else role
            return name, full_role
    except:
        return "UNKNOWN CANDIDATE", "N/A"

# --- MAIN LOGIC ---
def run_job_matching_demo(jd_file_object):
    if jd_file_object is None:
        raise gr.Error("Please upload a JD file.")

    uploaded_jd_path = jd_file_object.name
    yield "", gr.update(value="Processing..."), ""

    try:
        # 1. Xử lý JD
        print(f"Processing JD: {uploaded_jd_path}")
        jd_result = process_jd_query(uploaded_jd_path)
        
        jd_summary = jd_result.get('summary', '')
        jd_struct = jd_result.get('extracted', {})
        job_info = jd_struct.get("job_info", {})
        skills = jd_struct.get("skills", {})

        # --- FORMAT TEXT JD (Chuẩn bị nội dung) ---
        full_jd_text = f"""
### JD ANALYSIS REPORT
* **Job Title:** {job_info.get('job_title', 'N/A')}
* **Level / Exp:** {job_info.get('seniority_level', 'N/A')} (Min {job_info.get('min_years_experience', 0)} years)
* **Location:** {job_info.get('location', 'N/A')}

#### REQUIRED SKILLS
{', '.join(skills.get('required', []))}

#### PREFERRED SKILLS
{', '.join(skills.get('preferred', []))}
"""
        # --- TYPEWRITER EFFECT (Chạy chữ từng đoạn) ---
        stream_buffer = ""
        # Chạy nhanh hơn một chút (bước nhảy 3 ký tự) để không bị lag
        for i in range(0, len(full_jd_text), 3):
            stream_buffer = full_jd_text[:i+3]
            yield jd_summary, stream_buffer, "Searching..."
            time.sleep(0.005)
        
        # Đảm bảo hiện hết chữ cuối cùng
        yield jd_summary, full_jd_text, "Searching..."

        # 2. Chạy Search
        results = hybrid_search_v2(
            jd_query=jd_summary, 
            jd_struct=jd_struct,
            k_faiss=50, k_bm25=50, top_show=TOP_N
        )

        if not results:
            yield jd_summary, full_jd_text, "No suitable candidates found."
            return

        # 3. Xử lý Kết quả
        df = pd.DataFrame(results)
        meta_data = df['summary_file'].apply(get_metadata_from_json)
        df['Candidate Name'] = [x[0] for x in meta_data]
        df['Job'] = [x[1] for x in meta_data]
        df['CV File'] = df['pdf_path'].apply(os.path.basename)

        req_skills = set([s.lower().strip() for s in skills.get('required', [])])
        pref_skills = set([s.lower().strip() for s in skills.get('preferred', [])])
        all_jd_skills = req_skills.union(pref_skills)
        
        def find_common(cv_skills_list):
            if not isinstance(cv_skills_list, list): return ""
            found = []
            for s in cv_skills_list:
                s_lower = s.lower().strip()
                for target in all_jd_skills:
                    if target in s_lower or s_lower in target:
                        found.append(s)
                        break
            return ", ".join(sorted(list(set(found))))

        df['Common Skills'] = df['cv_skills_list'].apply(find_common)
        df['Match Score'] = df['final_score'].apply(lambda x: f"{x:.4f}")

        final_df = df[['Candidate Name', 'Job', 'CV File', 'Match Score', 'Common Skills']]
        html_table = convert_df_to_html(final_df, config.DRIVE_FOLDER_URL)
        
        yield jd_summary, full_jd_text, html_table

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"System Error: {str(e)}")

# --- UI CONFIGURATION (SOFT PASTEL PINK) ---
theme = gr.themes.Soft(
    primary_hue="pink",    
    neutral_hue="slate",
    text_size="sm",
    radius_size="sm"
).set(
    # Nút bấm: Hồng cam phấn (#ea7e8d - Mã bạn thích)
    button_primary_background_fill="#ea7e8d", 
    button_primary_background_fill_hover="#e11d48", 
    button_primary_text_color="#ffffff",
    block_title_text_color="#be185d",
)

# CSS Tùy chỉnh (Fix lỗi 2 thanh scroll + Màu sắc)
css_style = """
.gradio-container { font-family: 'Segoe UI', sans-serif; }
h1 { color: #9d174d !important; font-weight: 800 !important; text-transform: uppercase; letter-spacing: 1px; }

/* FIX LỖI 2 THANH SCROLL: Ẩn thanh cuộn của container ngoài */
.jd-analysis-box {
    height: 300px !important;
    border: 1px solid #fecdd3 !important; /* Viền hồng phấn */
    background-color: #fff1f2 !important; /* Nền hồng rất nhạt */
    border-radius: 8px !important;
    padding: 0 !important; /* Bỏ padding ngoài để scrollbar sát lề */
    overflow: hidden !important; /* Ẩn scrollbar thừa */
}

/* Chỉ cho phép scroll ở nội dung bên trong (Prose) */
.jd-analysis-box .prose {
    height: 100% !important;
    overflow-y: auto !important;
    padding: 15px !important;
}

/* Tùy chỉnh thanh cuộn cho đẹp (Chrome/Webkit) */
.jd-analysis-box .prose::-webkit-scrollbar { width: 6px; }
.jd-analysis-box .prose::-webkit-scrollbar-track { background: transparent; }
.jd-analysis-box .prose::-webkit-scrollbar-thumb { background: #fda4af; border-radius: 3px; }
.jd-analysis-box .prose::-webkit-scrollbar-thumb:hover { background: #f43f5e; }
"""

with gr.Blocks(theme=theme, css=css_style, title="Job Matching System") as demo:
    gr.Markdown("# JOB MATCHING SYSTEM")
    gr.Markdown("Professional CV Screening & Ranking.")
    
    with gr.Row():
        with gr.Column(scale=4):
            input_file = gr.File(label="Upload JD (PDF)", file_types=[".pdf"])
            btn = gr.Button("ANALYZE & MATCH", variant="primary")
        
        with gr.Column(scale=6):
            # Dùng Markdown và gán class CSS để fix scroll
            jd_display_box = gr.Markdown(
                label="JD Analysis", 
                elem_classes=["jd-analysis-box"],
                value="*Waiting for input...*"
            )
            
    result_area = gr.HTML(label="Ranking Results")
    hidden_summary = gr.Textbox(visible=False)

    btn.click(
        fn=run_job_matching_demo,
        inputs=[input_file],
        outputs=[hidden_summary, jd_display_box, result_area]
    )

if __name__ == "__main__":
    demo.launch()