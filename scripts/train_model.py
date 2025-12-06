import os, sys
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sklearn.model_selection import train_test_split

# --- 1. CẤU HÌNH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'artifacts', 'golden_dataset_v2.csv')
OUTPUT_MODEL_PATH = os.path.join(BASE_DIR, 'artifacts', 'my_fine_tuned_model')
CHART_PATH = os.path.join(BASE_DIR, 'artifacts', 'training_performance.png')

PRETRAINED_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BATCH_SIZE = 8       
NUM_EPOCHS = 5        
LEARNING_RATE = 2e-5

# --- HÀM PHỤ TRỢ ĐỂ SỬA LỖI DICT ---
def extract_score(score_obj):
    """
    Trích xuất điểm số float từ kết quả evaluator.
    Hỗ trợ cả version cũ (float) và mới (dict).
    """
    if isinstance(score_obj, dict):
        # Lấy giá trị 'spearman' nếu có, nếu không lấy giá trị đầu tiên
        if 'spearman' in score_obj:
            return score_obj['spearman']
        return list(score_obj.values())[0]
    return score_obj

def main():
    # --- 2. LOAD DỮ LIỆU ---
    if not os.path.exists(DATASET_PATH):
        return print("❌ Không tìm thấy file dataset!")

    print("📂 Đang đọc dữ liệu Golden Dataset V2...")
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=['score'])
    
    print(f"   -> Tổng số mẫu: {len(df)}")
    print("   -> Phân phối điểm số:", dict(df['score'].value_counts()))

    # Chia Train (80%) - Validation (20%)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_examples = [InputExample(texts=[r['jd_text'], r['cv_text']], label=float(r['score'])) for _, r in train_df.iterrows()]
    val_examples = [InputExample(texts=[r['jd_text'], r['cv_text']], label=float(r['score'])) for _, r in val_df.iterrows()]
    
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    evaluator = CECorrelationEvaluator.from_input_examples(val_examples, name='val_evaluator')

    # --- 3. KHỞI TẠO MODEL ---
    print(f"🤖 Tải model gốc: {PRETRAINED_MODEL}...")
    model = CrossEncoder(PRETRAINED_MODEL, num_labels=1)

    # --- 4. TRAINING LOOP ---
    print("\n🚀 BẮT ĐẦU HUẤN LUYỆN (Theo dõi từng Epoch)...")
    
    history = {
        "epoch": [],
        "spearman_score": []
    }

    # Đánh giá model trước khi train (Baseline)
    print("   👉 Đang kiểm tra model gốc...")
    raw_score = evaluator(model)
    baseline_score = extract_score(raw_score) 
    
    history["epoch"].append(0)
    history["spearman_score"].append(baseline_score)
    print(f"   Epoch 0 (Chưa học): Spearman = {baseline_score:.4f}")

    # Bắt đầu vòng lặp train
    for epoch in range(1, NUM_EPOCHS + 1):
        # Train 1 epoch
        model.fit(
            train_dataloader=train_dataloader,
            epochs=1,
            warmup_steps=int(len(train_dataloader) * 0.1),
            optimizer_params={'lr': LEARNING_RATE},
            show_progress_bar=True
        )
        
        # Đánh giá ngay lập tức
        raw_score = evaluator(model)
        current_score = extract_score(raw_score) 
        
        history["epoch"].append(epoch)
        history["spearman_score"].append(current_score)
        
        print(f"   ✅ Epoch {epoch}/{NUM_EPOCHS}: Spearman = {current_score:.4f} " 
              f"({'Tăng' if current_score > baseline_score else 'Giảm'})")

    # --- 5. LƯU MODEL ---
    print("\n💾 Đang lưu model final...")
    model.save(OUTPUT_MODEL_PATH)
    print(f"✅ Model đã lưu tại: {OUTPUT_MODEL_PATH}")

    # VẼ & LƯU BIỂU ĐỒ ---
    plt.figure(figsize=(10, 6))
    plt.plot(history["epoch"], history["spearman_score"], marker='o', linestyle='-', color='b', label='Validation Score')
    plt.title('Model Learning Progress (Knowledge Distillation)', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Spearman Correlation (Higher is Better)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(history["epoch"])
    
    for i, txt in enumerate(history["spearman_score"]):
        plt.annotate(f"{txt:.3f}", (history["epoch"][i], history["spearman_score"][i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.savefig(CHART_PATH)
    print(f"📊 Đã lưu biểu đồ huấn luyện tại: {CHART_PATH}")

if __name__ == "__main__":
    main()