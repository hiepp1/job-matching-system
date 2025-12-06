import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.core.ontology import skill_ontology

def test_ontology_power():
    print("\n🤖 --- TEST 1: KHẢ NĂNG ĐỌC & CHUẨN HÓA (Normalization) ---")
    print("Hệ thống có thể nhận diện các từ viết tắt, từ lóng không?")
    
    # Danh sách các từ khóa "bụi đời" thường gặp trong CV
    messy_skills = [
        "reactjs", "react.js", "React JS",  # Các kiểu viết React
        "k8s", "kube",                      # Kubernetes
        "js", "es6",                        # JavaScript
        "py", "python3",                    # Python
        "aws", "amazon web services",       # Cloud
        "postgres", "pg"                    # PostgreSQL
    ]
    
    print(f"\nInput Rác: {messy_skills}")
    print("-" * 50)
    
    normalized = []
    for s in messy_skills:
        norm = skill_ontology.normalize_skill(s)
        normalized.append(norm)
        print(f"✅ '{s}' \t---> '{norm}'")
        
    print("-" * 50)
    
    
    print("\n\n🤖 --- TEST 2: KHẢ NĂNG HIỂU QUAN HỆ (Relationships) ---")
    print("Hệ thống chấm điểm thế nào nếu không khớp từ khóa?")
    
    test_pairs = [
        ("MySQL", "PostgreSQL"),  # Cùng là SQL DB (Thay thế được)
        ("React", "Angular"),     # Cùng là Frontend (Thay thế được)
        ("React", "TypeScript"),  # Thường đi chung (Bổ trợ)
        ("Java", "Python"),       # Ngôn ngữ khác nhau (Ít liên quan)
        ("Docker", "Kubernetes"), # DevOps (Bổ trợ mạnh)
        ("HTML", "Machine Learning") # Không liên quan
    ]
    
    print(f"{'JD Cần':<15} | {'CV Có':<15} | {'Điểm':<5} | {'AI Hiểu Là'}")
    print("-" * 60)
    
    for s1, s2 in test_pairs:
        score = skill_ontology.check_relationship(s1, s2)
        
        meaning = "❌ Không liên quan"
        if score == 1.0: meaning = "🎯 Trùng khớp"
        elif score >= 0.6: meaning = "🤝 Bổ trợ (Complement)"
        elif score >= 0.4: meaning = "🔄 Thay thế (Alternative)"
        elif score > 0: meaning = "⚠️ Liên quan nhẹ"
            
        print(f"{s1:<15} | {s2:<15} | {score:<5} | {meaning}")

if __name__ == "__main__":
    # Sửa lại dòng trên thành:
    test_ontology_power()