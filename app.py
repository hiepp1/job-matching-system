## Import
import gradio as gr
import pandas as pd
import time
import os
import re
import json

import config
from src.llm_utils import process_jd_query
from src.search import hybrid_search_v2

## ====================== MAIN CODE ====================
def convert_df_to_html(df, drive_folder_url):
    """Tạo bảng HTML tùy chỉnh từ DataFrame và thêm link Google Drive."""
    html = """
    <style>
        .gradio-table { width: 100%; border-collapse: collapse; }
        .gradio-table th, .gradio-table td { 
            padding: 8px 12px; 
            border: 1px solid #e0e0e0; 
            text-align: left; 
            font-family: 'Times New Roman', serif;
            font-size: 0.9rem;
        }
        .gradio-table th { background-color: #f9f9f9; }
        .gradio-table tr:hover { background-color: #f1f1f1; }
        .gradio-table a { color: #0b57d0; text-decoration: none; }
        .gradio-table a:hover { text-decoration: underline; }
    </style>
    <table class='gradio-table'>
    """
    
    # Tạo header
    html += "<thead><tr>"
    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead>"
    
    # Tạo body
    html += "<tbody>"
    for _, row in df.iterrows():
        html += "<tr>"
        for col_name, val in row.items():
            if col_name == 'CV File':
                filename = val 
                html += f"<td><a href='{drive_folder_url}' target='_blank'>{filename}</a></td>"
            else:
                html += f"<td>{val}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    return html

def strip_accents(text):
    """Xóa dấu tiếng Việt."""
    s = text
    s = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', s)
    s = re.sub(r'[ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ]', 'A', s)
    s = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', s)
    s = re.sub(r'[ÈÉẸẺẼÊỀẾỆỂỄ]', 'E', s)
    s = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', s)
    s = re.sub(r'[ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ]', 'O', s)
    s = re.sub(r'[ìíịỉĩ]', 'i', s)
    s = re.sub(r'[ÌÍỊỈĨ]', 'I', s)
    s = re.sub(r'[ùúụủũưừứựửữ]', 'u', s)
    s = re.sub(r'[ÙÚỤỦŨƯỪỨỰỬỮ]', 'U', s)
    s = re.sub(r'[ỳýỵỷỹ]', 'y', s)
    s = re.sub(r'[ỲÝỴỶỸ]', 'Y', s)
    s = re.sub(r'[đ]', 'd', s)
    s = re.sub(r'[Đ]', 'D', s)
    return s

def normalize_vietnamese_name(name):
    """Chuẩn hóa tên: Xóa dấu và viết HOA."""
    if not name or name == "N/A" or name == "Error N/A":
        return "N/A"
    name_no_accents = strip_accents(name)
    return name_no_accents.upper()

def get_name_from_json(summary_filename):
    """Lấy tên ứng viên từ file JSON."""
    if not summary_filename or not isinstance(summary_filename, str):
        return "N/A"
    try:
        base_name = os.path.splitext(summary_filename)[0]
        json_filepath = os.path.join(config.CV_JSON_FOLDER, f"{base_name}.json")
        with open(json_filepath, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        return json_data.get("name", "Unknown Name")
    except Exception as e:
        return "Error N/A"

def norm_set(lst):
    """Chuẩn hóa list skills thành một set."""
    return set([re.sub(r'[^a-z0-9]', '', s.lower()).strip() for s in lst if s])

def find_common_skills(cv_skills_list, jd_skills_set):
    """Tìm các skill chung giữa CV và JD."""
    if not cv_skills_list: return "N/A"
    cv_skills_set = norm_set(cv_skills_list)
    common_skills = jd_skills_set & cv_skills_set
    if not common_skills: return "None"
    return ", ".join(sorted(list(common_skills)))


# DEMO FUNCTION
def run_job_matching_demo(jd_file_object):

    TOP_N_TO_DISPLAY = config.TOP_N_TO_DISPLAY

    # --- A. Kiểm tra Input ---
    if jd_file_object is None:
        raise gr.Error("Lỗi: Bạn chưa tải lên file JD.")
        
    uploaded_jd_path = jd_file_object.name
    
    if not uploaded_jd_path.lower().endswith('.pdf'):
        raise gr.Error("Lỗi: File tải lên phải là dạng .pdf. Vui lòng thử lại.")

    print(f"Đã nhận file: {uploaded_jd_path}")
    print("------------------------------------------------------------")
    
    # Xóa kết quả cũ (nếu có)
    yield "", "", "Processing..." 

    # --- B. Chạy Pipeline ---
    try:
        print("Đang xử lý JD...")
        jd_result = process_jd_query(uploaded_jd_path) 
        jd_summary = jd_result['summary']
        jd_struct = jd_result.get('extracted', {})

        # --- C. TYPEWRITER JD SUMMARY ---
        output_jd_summary = jd_summary
        jd_skills_list = jd_struct.get("skills", [])
        output_jd_skills = ", ".join(jd_skills_list) if jd_skills_list else "No skills extracted."

        summary_stream = ""
        for char in output_jd_summary:
            summary_stream += char
            time.sleep(0.005) 
            yield summary_stream, "", "Processing..."

        # --- D. TYPEWRITER JD SKILLS ---
        skills_stream = ""
        for char in output_jd_skills:
            skills_stream += char
            time.sleep(0.005) 
            yield summary_stream, skills_stream, "Processing..."

        # --- E. Chạy tìm kiếm (Sau khi đã xong typewriter) ---
        print(f"\nĐang tìm kiếm và xếp hạng {TOP_N_TO_DISPLAY} kết quả hàng đầu...")
        results = hybrid_search_v2(
            jd_summary, 
            jd_struct, 
            k_faiss=100, 
            k_bm25=200, 
            top_show=TOP_N_TO_DISPLAY
        )

        if not results:
            print("Không tìm thấy kết quả nào.")
            yield summary_stream, skills_stream, "Không tìm thấy kết quả phù hợp."
            return

        # --- F. Xử lý Output thành DataFrame ---
        print(f"===== TOP {TOP_N_TO_DISPLAY} RESULTS  =====")
        df = pd.DataFrame(results)
        df = df.sort_values(by='rank')

        # Đổi tên cột
        df = df.rename(columns={
            'rank': 'Rank',
            'final_score': 'Match Score', 
            'skill': 'Skill',
            'industry_match': 'Job'
        })
        
        # Làm tròn cột điểm
        df['Match Score'] = df['Match Score'].round(2)
        df['Skill'] = df['Skill'].round(2)
        
        df['Job'] = df['Job'].apply(lambda x: "Match" if x == 1.0 else "Not Match")

        # Lấy tên
        df['Candidate Name'] = df['summary_file'].apply(get_name_from_json)
        df['Candidate Name'] = df['Candidate Name'].apply(normalize_vietnamese_name)
        
        df['CV File'] = df['pdf_path'].apply(os.path.basename)

        # Tính Common Skills
        jd_skills_set = norm_set(jd_struct.get("skills", []))
        df['Common Skills'] = df['cv_skills_list'].apply(lambda cv_list: find_common_skills(cv_list, jd_skills_set))

        # Chọn cột
        columns_to_show = ['CV File', 'Candidate Name', 'Match Score', 'Skill', 'Job', 'Common Skills']
        df_display = df[columns_to_show] 
        
        # --- G. CHUYỂN DF SANG HTML VÀ TRẢ VỀ ---
        final_html_table = convert_df_to_html(df_display, config.DRIVE_FOLDER_URL)
        
        yield summary_stream, skills_stream, final_html_table

    except Exception as e:
        print(f"\n❌ ĐÃ XẢY RA LỖI TRONG QUÁ TRÌNH XỬ LÝ: {e}")
        raise gr.Error(f"Lỗi xử lý: {e}. Vui lòng kiểm tra file PDF hoặc API key.")
    
theme = gr.themes.Soft(
    font=["Times New Roman", "serif"],
    font_mono=["IBM Plex Mono", "monospace"],
    text_size=gr.themes.sizes.text_sm,
)

with gr.Blocks(theme=theme) as demo_ui:

    gr.Markdown("## Job Matching ")
    gr.Markdown("Upload a Job Description file (pdf). System will returns top 10 match CV")
    
    with gr.Row():
        input_file = gr.File(label="Upload a JD", file_types=[".pdf"])
    
    with gr.Row():
        submit_button = gr.Button("Submit", variant="primary")

    with gr.Row():
        output_jd_summary = gr.Textbox(label="JD Summary", lines=5)
    with gr.Row():
        output_jd_skills = gr.Textbox(label="JD Skills", lines=3)
    
    with gr.Row():
        output_html_table = gr.HTML(
            label=f"Top {config.TOP_N_TO_DISPLAY} CV Phù Hợp"
        )
    
    # Logic
    submit_button.click(
        fn=run_job_matching_demo,
        inputs=input_file,
        outputs=[output_jd_summary, output_jd_skills, output_html_table]
    )

if __name__ == "__main__":
    print("Đang tải các index...")
    # (Bạn có thể thêm các hàm pre-load index ở đây nếu muốn)
    print("Khởi động Gradio UI...")
    demo_ui.launch(debug=True) # Bỏ share=True khi chạy local