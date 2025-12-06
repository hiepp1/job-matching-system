import gradio as gr
import pandas as pd
import time
import os
import json
import base64
from dotenv import load_dotenv

load_dotenv()

import config
from src.core.llm import process_jd_query
from src.core.search import hybrid_search_v2
from src.verification import verify_candidate_with_ai, generate_verification_html

try:
    from src.core.ontology import skill_ontology
except ImportError:
    from src.core.search import skill_ontology

TOP_N = 15

# Global session state
CURRENT_SEARCH_RESULTS = []
CURRENT_JD_STRUCT = {}

# --- HTML TABLE ---
def convert_df_to_html(df, drive_folder_url):
    html = """
    <style>
        .pro-table { width: 100%; border-collapse: collapse; font-family: 'Segoe UI', sans-serif; font-size: 0.95rem; color: #333; background-color: #fff; border: 1px solid #fce7f3; }
        .pro-table th { text-align: left; padding: 12px 15px; border-bottom: 2px solid #f472b6; background-color: #fdf2f8; color: #9d174d; font-weight: 700; text-transform: uppercase; font-size: 0.85rem; }
        .pro-table td { padding: 12px 15px; border-bottom: 1px solid #f3f4f6; vertical-align: top; }
        .pro-table tr:hover { background-color: #fff1f2; } 
        .candidate-name { font-weight: 700; font-size: 1rem; color: #1f2937; margin-bottom: 4px; }
        .candidate-role { font-size: 0.85rem; color: #64748b; margin-bottom: 6px; }
        .score-val { font-weight: 800; font-size: 1.1rem; color: #be185d; }
        .skill-tag { display: inline-block; border: 1px solid #e2e8f0; color: #475569; padding: 3px 8px; border-radius: 4px; font-size: 0.8rem; margin: 2px; background-color: #f8fafc; font-weight: 500; }
        .skill-match { border-color: #fbcfe8; background-color: #fdf2f8; color: #be185d; }
        .no-skill { color: #94a3b8; font-style: italic; font-size: 0.85rem; }
        .cv-link a { color: #db2777; text-decoration: none; font-size: 0.75rem; font-weight: 600; border: 1px solid #fbcfe8; padding: 3px 8px; border-radius: 4px; background-color: #fff; transition: all 0.2s; }
        .cv-link a:hover { background-color: #fce7f3; border-color: #db2777; }
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

        html += f"""
        <tr>
            <td>
                <div class="candidate-name">{row['Candidate Name']}</div>
                <div class="candidate-role">{row['Job']}</div>
            </td>
            <td><div class="score-val">{row['Match Score']}</div></td>
            <td>{skills_html}</td>
        </tr>
        """
    html += "</tbody></table>"
    return html

def get_metadata_from_json(summary_filename):
    try:
        base_name = os.path.splitext(summary_filename)[0]
        json_path = os.path.join(config.CV_JSON_FOLDER, f"{base_name}.json")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            profile = data.get("candidate_profile", {})
            metrics = data.get("metrics", {})

            name = profile.get("name", "Unknown").upper()
            role = profile.get('role_focus', 'N/A')
            level = profile.get('seniority_level', '')
            full_role = f"{role} • {level}" if level else role
            exp = metrics.get('years_experience', 0)

            return name, full_role, exp
    except:
        return "UNKNOWN CANDIDATE", "N/A", 0

def _pdf_to_base64(pdf_path: str) -> str:
    with open(pdf_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_clean_filename(path_str):
    if not path_str: return ""
    normalized_path = path_str.replace('\\', '/')
    return normalized_path.split('/')[-1]

def preview_cv_inline(rank_idx):
    global CURRENT_SEARCH_RESULTS
    try:
        idx = int(rank_idx) - 1
        if not CURRENT_SEARCH_RESULTS or idx < 0 or idx >= len(CURRENT_SEARCH_RESULTS):
            return "<div>⚠️ Please run 'Analyze & Match' and choose a valid rank.</div>", "Selected Candidate Score: N/A"
        
        candidate = CURRENT_SEARCH_RESULTS[idx]

        raw_path = candidate.get('pdf_path', '')
        pdf_name = get_clean_filename(raw_path)
        full_path = os.path.join(config.CV_FOLDER, pdf_name)
        
        score = candidate.get('final_score', 0.0)
        percent_score = f"{score * 100:.1f}%"

        if not os.path.exists(full_path):
            return f"<div>❌ PDF not found at: {full_path}</div>", "N/A"
        
        pdf_b64 = _pdf_to_base64(full_path)
        html = f"""
        <div style="border:1px solid #e5e7eb; border-radius:8px; overflow:hidden;">
            <iframe src="data:application/pdf;base64,{pdf_b64}" width="100%" height="700px" style="border:0;"></iframe>
        </div>
        """
        return html, f"### 🎯 Selected Candidate Score: {percent_score}"
    except Exception as e:
        return f"<div>❌ Error: {str(e)}</div>", "Selected Candidate Score: N/A"

# --- Verification ---
def analyze_candidate_details(rank_idx):
    global CURRENT_SEARCH_RESULTS, CURRENT_JD_STRUCT
    
    if not CURRENT_SEARCH_RESULTS: 
        return "⚠️ Please run 'Analyze & Match' first."
    
    try:
        idx = int(rank_idx) - 1
        if idx < 0 or idx >= len(CURRENT_SEARCH_RESULTS):
            return "⚠️ Invalid rank number."

        candidate = CURRENT_SEARCH_RESULTS[idx]
        cv_summary_text = candidate.get("summary_snippet", "")
        report_data = verify_candidate_with_ai(CURRENT_JD_STRUCT, cv_summary_text)
        return generate_verification_html(report_data)
    except Exception as e:
        return f"❌ Error: {str(e)}"

# --- MAIN LOGIC ---
def run_job_matching(jd_file_object):
    global CURRENT_SEARCH_RESULTS, CURRENT_JD_STRUCT

    if jd_file_object is None:
        raise gr.Error("Please upload a JD file.")

    uploaded_jd_path = jd_file_object.name
    yield "", gr.update(value="Processing..."), "", ""

    try:
        # 1. Process JD
        jd_result = process_jd_query(uploaded_jd_path)
        jd_summary = jd_result.get('summary', '')
        jd_struct = jd_result.get('extracted', {})
        CURRENT_JD_STRUCT = jd_struct
        
        job_info = jd_struct.get("job_info", {})
        skills = jd_struct.get("skills", {})

        # Format JD text
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
        
        typed_text = ""
        for ch in full_jd_text:
            typed_text += ch
            yield jd_summary, typed_text, "Searching...", ""
            time.sleep(0.005)      
        
        yield jd_summary, full_jd_text, "Searching...", ""

        # 2. Run Search
        results = hybrid_search_v2(
            jd_query=jd_summary, 
            jd_struct=jd_struct,
            k_faiss=50, k_bm25=50, top_show=TOP_N
        )

        print(f"\n{'='*130}")
        print(f"🚀 KẾT QUẢ CHI TIẾT TOP {TOP_N} (Kèm Phạt/Thưởng)")
        print(f"{'RK':<3} | {'NAME':<25} | {'FINAL':<6} | {'SEM':<6} | {'SKL':<6} | {'AI':<6} | {'SEN-PEN':<7} | {'ENG-PEN':<7} | {'ACH-BON':<7}")
        print(f"{'-'*130}")

        for res in results:
            name, _, _ = get_metadata_from_json(res.get('summary_file'))
            if name in ["N/A", "Error N/A", "Unknown Name", "UNKNOWN CANDIDATE"]:
                name = os.path.basename(res.get('pdf_path', 'Unknown'))
            
            # Lấy các điểm số thành phần
            final = res.get('final_score', 0)
            sem = res.get('semantic', 0)
            skill = res.get('skill', 0)
            ai = res.get('cross_score', 0)
            
            # Lấy thông tin Phạt/Thưởng 
            factors = res.get('factors', {}) 
            sen_pen = factors.get('sen_penalty', 1.0) # Mặc định 1.0 (Không phạt)
            eng_pen = factors.get('eng_penalty', 1.0)
            ach_bon = factors.get('ach_bonus', 1.0)   # Mặc định 1.0 (Không thưởng)

            cv_filename = os.path.basename(res.get('pdf_path', '') or '')
            cv_id = os.path.splitext(cv_filename)[0]
            display_name = f"{name} ({cv_id})" if cv_id else name

            print(f"#{res['rank']:<2} | {display_name[:25]:<25} | {final:.4f} | {sem:.4f} | {skill:.4f} | {ai:.4f} | {sen_pen:.4f}  | {eng_pen:.4f}  | {ach_bon:.4f}")
        
        print(f"{'='*130}\n")
        
        CURRENT_SEARCH_RESULTS = results

        if not results:
            yield jd_summary, full_jd_text, "No suitable candidates found.", ""
            return

        # 3. Build results table
        df = pd.DataFrame(results)

        meta_data = df['summary_file'].apply(get_metadata_from_json)
        df['Candidate Name'] = [x[0] for x in meta_data]
        df['Job'] = [x[1] for x in meta_data]
        df['Experience'] = [x[2] for x in meta_data]

        req_skills = set([s.lower().strip() for s in skills.get('required', [])])
        pref_skills = set([s.lower().strip() for s in skills.get('preferred', [])])
        all_jd_skills = req_skills.union(pref_skills)
        
        def find_common(cv_skills_list):
            if not isinstance(cv_skills_list, list): return ""
            found = set()
            for s in cv_skills_list:
                for jd_skill in all_jd_skills:
                    if skill_ontology.check_relationship(s, jd_skill) >= 0.9:
                        found.add(s)
                        break
            return ", ".join(sorted(list(found)))

        df['Common Skills'] = df['cv_skills_list'].apply(find_common)
        df['Match Score'] = df['final_score'].apply(lambda x: f"{x * 100:.1f}%")

        final_df = df[['Candidate Name', 'Job', 'Experience', 'Match Score', 'Common Skills']]
        html_table = convert_df_to_html(final_df, config.DRIVE_FOLDER_URL)
        
        default_preview = "<div style='color:#6b7280;'>Select a rank and click 'Preview CV' to view inline.</div>"
        yield jd_summary, full_jd_text, html_table, default_preview

    except Exception as e:
        print(f"Error: {e}")
        raise gr.Error(f"System Error: {str(e)}")

# --- UI ---
theme = gr.themes.Soft(
    primary_hue="pink",    
    neutral_hue="slate",
    text_size="sm",
    radius_size="sm"
).set(
    button_primary_background_fill="#ea7e8d", 
    button_primary_background_fill_hover="#e11d48", 
    button_primary_text_color="#ffffff",
    block_title_text_color="#be185d",
)

css_style = """
.gradio-container { font-family: 'Segoe UI', sans-serif; }
h1 { color: #9d174d !important; font-weight: 800 !important; text-transform: uppercase; letter-spacing: 1px; }
.jd-analysis-box {
    height: 300px !important;
    border: 1px solid #fecdd3 !important;
    background-color: #fff1f2 !important;
    border-radius: 8px !important;
    padding: 0 !important;
    overflow: hidden !important;
}
.jd-analysis-box .prose { height: 100% !important; overflow-y: auto !important; padding: 15px !important; }
"""

with gr.Blocks(theme=theme, css=css_style, title="Job Matching System") as demo:
    gr.Markdown("# AI JOB MATCHING SYSTEM")

    with gr.Row():
        # Sidebar Tabs
        with gr.Column(scale=2):
            tab_selector = gr.Radio(
                choices=["Matching & Ranking", "Detailed Analysis", "Preview CV"],
                label="", 
                value="Matching & Ranking"
            )

            # Inputs grouped by tab
            input_file = gr.File(label="Upload JD (PDF)", file_types=[".pdf"], visible=True)
            rank_input = gr.Number(label="Rank #", value=1, precision=0, visible=False)

            # Buttons grouped by tab
            btn_match = gr.Button("ANALYZE & MATCH", variant="primary", visible=True)
            btn_preview = gr.Button("PREVIEW CV", variant="secondary", visible=False)
            btn_analysis = gr.Button("View Detailed Analysis", variant="secondary", visible=False)

        with gr.Column(scale=8):
            with gr.Group(visible=True) as matching_group:
                jd_display_box = gr.Markdown(label="JD Analysis", elem_classes=["jd-analysis-box"])
                result_area = gr.HTML(label="Ranking Results")

            with gr.Group(visible=False) as analysis_group:
                analysis_output = gr.HTML(label="Verification Report")

            with gr.Group(visible=False) as preview_group:
                selected_score_md = gr.Markdown("### 🎯 Selected Candidate Score: N/A")
                pdf_viewer_html = gr.HTML(
                    label="CV Preview",
                    value="<div style='color:#6b7280;'>Select a rank and click 'PREVIEW CV'.</div>"
                )

    # --- Tab Switching Logic ---
    def switch_tab(tab_name):
        return (
            gr.update(visible=tab_name == "Matching & Ranking"),  # matching_group
            gr.update(visible=tab_name == "Detailed Analysis"),   # analysis_group
            gr.update(visible=tab_name == "Preview CV"),          # preview_group
            gr.update(visible=(tab_name == "Matching & Ranking")), # input_file
            gr.update(visible=(tab_name in ["Detailed Analysis", "Preview CV"])), # rank_input
            gr.update(visible=(tab_name == "Matching & Ranking")), # btn_match
            gr.update(visible=(tab_name == "Preview CV")),         # btn_preview
            gr.update(visible=(tab_name == "Detailed Analysis"))   # btn_analysis
        )

    tab_selector.change(
        fn=switch_tab,
        inputs=[tab_selector],
        outputs=[
            matching_group, analysis_group, preview_group,
            input_file, rank_input,
            btn_match, btn_preview, btn_analysis
        ]
    )

    def switch_tab(tab_name):
        return (
            gr.update(visible=tab_name == "Matching & Ranking"),  # matching_group
            gr.update(visible=tab_name == "Detailed Analysis"),   # analysis_group
            gr.update(visible=tab_name == "Preview CV"),          # preview_group
            gr.update(visible=(tab_name == "Matching & Ranking")), # btn_match
            gr.update(visible=(tab_name == "Preview CV")),         # btn_preview
            gr.update(visible=(tab_name == "Detailed Analysis"))   # btn_analysis
        )

    tab_selector.change(
        fn=switch_tab,
        inputs=[tab_selector],
        outputs=[matching_group, analysis_group, preview_group,
                 btn_match, btn_preview, btn_analysis]
    )

    # EVENTS
    btn_match.click(
        fn=run_job_matching,
        inputs=[input_file],
        outputs=[jd_display_box, jd_display_box, result_area, pdf_viewer_html]
    )

    btn_preview.click(
        fn=preview_cv_inline,
        inputs=[rank_input],
        outputs=[pdf_viewer_html, selected_score_md]
    )

    btn_analysis.click(
        fn=analyze_candidate_details,
        inputs=[rank_input],
        outputs=[analysis_output]
    )

if __name__ == "__main__":
    print("Server is running...")
    demo.queue().launch(
        server_name="0.0.0.0", 
        server_port=7860,       
        share=False,           
        debug=True             
    )



# import streamlit as st
# import pandas as pd
# import os
# import json
# import base64
# import config

# from src.core.llm import process_jd_query
# from src.core.search import hybrid_search_v2
# from src.verification import verify_candidate_with_ai, generate_verification_html
# from src.core.ontology import skill_ontology

# st.set_page_config(page_title="AI Job Matching System", layout="wide")
# st.title("AI Job Matching System")

# TOP_N = 15

# # ---------------- Session state ---------------- #
# if "jd_struct" not in st.session_state:
#     st.session_state.jd_struct = {}
# if "search_results" not in st.session_state:
#     st.session_state.search_results = []
# if "selected_rank" not in st.session_state:
#     st.session_state.selected_rank = 1

# # ---------------- Helpers ---------------- #
# def pdf_to_base64(pdf_path: str) -> str:
#     with open(pdf_path, "rb") as f:
#         return base64.b64encode(f.read()).decode("utf-8")

# def render_pdf_inline(pdf_path: str, height: int = 700):
#     if not os.path.exists(pdf_path):
#         st.error(f"PDF not found: {pdf_path}")
#         return
#     pdf_b64 = pdf_to_base64(pdf_path)
#     html = f"""
#     <iframe src="data:application/pdf;base64,{pdf_b64}" width="100%" height="{height}px" style="border:0;"></iframe>
#     """
#     st.components.v1.html(html, height=height + 20, scrolling=True)

# # ---------------- Upload JD ---------------- #
# jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])

# if jd_file and st.button("Analyze & Match"):
#     jd_path = os.path.join("temp", jd_file.name)
#     os.makedirs("temp", exist_ok=True)
#     with open(jd_path, "wb") as f:
#         f.write(jd_file.read())

#     jd_result = process_jd_query(jd_path)
#     st.session_state.jd_struct = jd_result.get("extracted", {})
#     jd_summary = jd_result.get("summary", "")

#     # Run search once
#     with st.spinner("Searching candidates..."):
#         results = hybrid_search_v2(
#             jd_query=jd_summary,
#             jd_struct=st.session_state.jd_struct,
#             k_faiss=50,
#             k_bm25=50,
#             top_show=TOP_N,
#         )
#         st.session_state.search_results = results

#     st.subheader("📑 JD Analysis")
#     st.markdown(jd_summary)

#     if results:
#         df = pd.DataFrame(results)
#         df["Match Score"] = df["final_score"].apply(lambda x: f"{x*100:.1f}%")
#         st.subheader("🏆 Candidate Ranking")
#         st.dataframe(df[["rank", "document_id", "Match Score", "cv_skills_list"]])

# # ---------------- Tabs ---------------- #
# if st.session_state.search_results:
#     tab1, tab2 = st.tabs(["📄 CV Preview", "🔍 Detailed Analysis"])

#     with tab1:
#         st.session_state.selected_rank = st.number_input(
#             "Select candidate rank", min_value=1, max_value=len(st.session_state.search_results), step=1,
#             value=st.session_state.selected_rank
#         )
#         if st.button("Preview CV"):
#             candidate = st.session_state.search_results[st.session_state.selected_rank - 1]
#             st.markdown(f"### 🎯 Candidate Score: {candidate['final_score']*100:.1f}%")
#             st.write("Skills:", ", ".join(candidate.get("cv_skills_list", [])))
#             pdf_path = candidate.get("pdf_path", "")
#             render_pdf_inline(pdf_path)

#     with tab2:
#         if st.button("Run Detailed Analysis"):
#             candidate = st.session_state.search_results[st.session_state.selected_rank - 1]
#             report = verify_candidate_with_ai(st.session_state.jd_struct, candidate.get("summary_snippet", ""))
#             st.markdown(generate_verification_html(report), unsafe_allow_html=True)
