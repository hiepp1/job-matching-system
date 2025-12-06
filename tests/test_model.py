import os
import pandas as pd
import numpy as np
from sentence_transformers import CrossEncoder
from scipy.stats import spearmanr

# --- CẤU HÌNH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'artifacts', 'golden_dataset.csv')
NEW_MODEL_PATH = os.path.join(BASE_DIR, 'artifacts', 'my_fine_tuned_model')
OLD_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def main():
    # 1. Đọc dữ liệu kiểm tra
    if not os.path.exists(DATASET_PATH):
        print("❌ Không tìm thấy file dữ liệu golden_dataset.csv")
        return
    
    print("📂 Đang đọc dữ liệu Golden Dataset...")
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=['score'])
    
    # Tạo list các cặp (JD, CV) để đưa vào model
    eval_pairs = [[row['jd_text'], row['cv_text']] for _, row in df.iterrows()]
    gold_scores = df['score'].tolist()
    
    print(f"   -> Số lượng mẫu kiểm tra: {len(df)}")

    # -------------------------------------------------------
    # 2. Đánh giá Model CŨ (Base Model)
    # -------------------------------------------------------
    print(f"\n🤖 Đang test Model CŨ ({OLD_MODEL_NAME})...")
    model_old = CrossEncoder(OLD_MODEL_NAME)
    scores_old = model_old.predict(eval_pairs)
    
    # Tính độ tương quan (Spearman)
    corr_old, _ = spearmanr(gold_scores, scores_old)
    print(f"   👉 Spearman Correlation (Độ hiểu ý Gemini): {corr_old:.4f}")

    # -------------------------------------------------------
    # 3. Đánh giá Model MỚI (Fine-tuned Model)
    # -------------------------------------------------------
    if not os.path.exists(NEW_MODEL_PATH):
        print(f"\n❌ Không tìm thấy Model Mới tại {NEW_MODEL_PATH}. Bạn đã train chưa?")
        return

    print(f"\n🚀 Đang test Model MỚI (Fine-tuned)...")
    model_new = CrossEncoder(NEW_MODEL_PATH)
    scores_new = model_new.predict(eval_pairs)
    
    # Tính độ tương quan
    corr_new, _ = spearmanr(gold_scores, scores_new)
    print(f"   👉 Spearman Correlation (Độ hiểu ý Gemini): {corr_new:.4f}")

    # -------------------------------------------------------
    # 4. So sánh & Kết luận
    # -------------------------------------------------------
    print("\n" + "="*40)
    print("📊 KẾT QUẢ SO SÁNH")
    print("="*40)
    print(f"Model Cũ: {corr_old:.4f}")
    print(f"Model Mới: {corr_new:.4f}")
    
    improvement = (corr_new - corr_old) * 100
    if corr_new > corr_old:
        print(f"✅ THÀNH CÔNG! Model mới thông minh hơn model cũ {improvement:.2f} điểm.")
    else:
        print(f"⚠️ CẢNH BÁO: Model mới không tốt hơn. Cần kiểm tra lại dữ liệu train.")

    # -------------------------------------------------------
    # 5. Show vài ví dụ thực tế (The Eye Test)
    # -------------------------------------------------------
    df['score_old'] = scores_old
    df['score_new'] = scores_new
    
    print("\n🔍 --- VÍ DỤ CỤ THỂ (Sự khác biệt) ---")
    # Lọc ra những ca mà model mới chấm khác biệt lớn so với model cũ
    df['diff'] = abs(df['score_new'] - df['score_old'])
    top_diff = df.sort_values(by='diff', ascending=False).head(5)
    
    for idx, row in top_diff.iterrows():
        print(f"\n[Mẫu #{idx}]")
        print(f"🎯 Gemini chấm: {row['score']}")
        print(f"❌ Model Cũ đoán: {row['score_old']:.4f}")
        print(f"✅ Model Mới đoán: {row['score_new']:.4f}")
        print(f"💡 Chênh lệch: {abs(row['score_new'] - row['score']):.4f} (Mới) vs {abs(row['score_old'] - row['score']):.4f} (Cũ)")

if __name__ == "__main__":
    main()