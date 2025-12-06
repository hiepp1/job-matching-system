import json
from src.core.llm import call_smart_verification, PROMPT_VERIFY_MATCH, call_smart_json

def verify_candidate_with_ai(jd_struct, cv_content):
    """
    Chuẩn bị dữ liệu và gọi AI để so sánh.
    """
    checklist = jd_struct.get("requirements_checklist", [])    

    if checklist:

        jd_text_list = [f"[{item.get('category', 'Requirement')}] {item.get('content', '')}" for item in checklist]
    else:

        print("⚠️ Warning: JD missing 'requirements_checklist'. Using fallback fields.")
        qualifications = jd_struct.get("qualifications", [])
        req_skills = jd_struct.get("skills", {}).get("required", [])
        job_info = jd_struct.get("job_info", {})
        
        jd_text_list = []
        if job_info.get("min_years_experience"):
            jd_text_list.append(f"[Experience] At least {job_info['min_years_experience']} years experience.")
        if qualifications:
            jd_text_list.extend([f"[Education] {q}" for q in qualifications])
        if req_skills:
            jd_text_list.append(f"[Tech Stack] Proficient in: {', '.join(req_skills[:7])}")

    jd_text_combined = "\n".join(jd_text_list)

    prompt_filled = PROMPT_VERIFY_MATCH.replace("{jd_requirements}", jd_text_combined).replace("{cv_profile}", cv_content)
        
    from src.core.llm import call_smart_json 
    from src.core.llm import call_smart_verification

    verification_result = call_smart_verification(jd_text_combined, cv_content)

    if isinstance(verification_result, dict) and "checks" in verification_result:
        return verification_result["checks"]
    elif isinstance(verification_result, list):
        return verification_result
    return []

def generate_verification_html(report):
    """Render HTML giống LinkedIn"""
    if not report: return "<p>Could not generate analysis.</p>"
    
    html = "<div style='font-family: Segoe UI, sans-serif; font-size: 0.95rem;'>"
    
    match_count = sum(1 for r in report if r['status'] == 'match')
    total = len(report)
    
    html += f"<div style='margin-bottom: 15px; font-weight: bold; color: #333;'>Matches {match_count} of {total} requirements:</div>"
    
    for item in report:
        status = item.get('status', 'unknown')
        reason = item.get('reason', '')
        req = item.get('requirement', '')
        
        if status == "match":
            icon = "✅"
            color = "#166534" 
            bg = ""
        elif status == "partial":
            icon = "⚠️" 
            color = "#ca8a04" 
            bg = "#fefce8"
        else: # missing/unknown
            icon = "❓"
            color = "#525252" 
            bg = ""

        html += f"""
        <div style='margin-bottom: 12px; padding: 8px; border-radius: 6px; background: {bg};'>
            <div style='display: flex; align_items: start;'>
                <span style='margin-right: 10px; font-size: 1.1rem;'>{icon}</span>
                <div>
                    <div style='color: #1f2937; font-weight: 500;'>{req}</div>
                    <div style='color: {color}; font-size: 0.85rem; margin-top: 2px;'>{reason}</div>
                </div>
            </div>
        </div>
        """
    html += "</div>"
    return html