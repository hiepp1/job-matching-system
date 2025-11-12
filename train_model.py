import os
import pandas as pd
import math
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sklearn.model_selection import train_test_split

# --- CẤU HÌNH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'artifacts', 'golden_dataset.csv')
OUTPUT_MODEL_PATH = os.path.join(BASE_DIR, 'artifacts', 'my_fine_tuned_model')

# Model gốc (chưa học) - Dùng bản MiniLM cho nhẹ
PRETRAINED_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Tham số huấn luyện
BATCH_SIZE = 8        # CPU thì để batch nhỏ thôi
NUM_EPOCHS = 5        # Số vòng train
LEARNING_RATE = 2e-5  # Tốc độ học

def main():
    # 1. Đọc dữ liệu
    if not os.path.exists(DATASET_PATH):
        print("❌ Không tìm thấy file golden_dataset.csv!")
        return

    print("📂 Đang đọc dữ liệu...")
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=['score'])
    print(f"   -> Tổng số mẫu dữ liệu: {len(df)}")

    # 2. Chia tập Train (80%) và Validation (20%)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # 3. Chuyển đổi dữ liệu
    train_examples = []
    for _, row in train_df.iterrows():
        train_examples.append(InputExample(
            texts=[row['jd_text'], row['cv_text']], 
            label=float(row['score'])
        ))
        
    # 4. Tạo DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

    # 5. Tạo Evaluator
    val_samples = []
    for _, row in val_df.iterrows():
        val_samples.append(InputExample(
            texts=[row['jd_text'], row['cv_text']], 
            label=float(row['score'])
        ))
    
    evaluator = CECorrelationEvaluator.from_input_examples(val_samples, name='val_evaluator')

    # 6. Tải Model
    print(f"🤖 Đang tải model gốc: {PRETRAINED_MODEL}...")
    model = CrossEncoder(PRETRAINED_MODEL, num_labels=1) 

    # 7. Bắt đầu Train
    print("🚀 Bắt đầu huấn luyện (Fine-tuning)...")
    warmup_steps = math.ceil(len(train_dataloader) * NUM_EPOCHS * 0.1) 

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=NUM_EPOCHS,
        warmup_steps=warmup_steps,
        output_path=OUTPUT_MODEL_PATH, # Lưu checkpoint tốt nhất vào đây
        optimizer_params={'lr': LEARNING_RATE},
        show_progress_bar=True,
        save_best_model=True # Quan trọng: Chỉ lưu model tốt nhất
    )

    # --- QUAN TRỌNG: LƯU MODEL THỦ CÔNG LẦN CUỐI ---
    # Đảm bảo dù thế nào cũng có file config.json và pytorch_model.bin
    print("💾 Đang lưu model lần cuối...")
    model.save(OUTPUT_MODEL_PATH)

    print(f"\n✅ HUẤN LUYỆN HOÀN TẤT!")
    print(f"📂 Model mới đã được lưu tại: {OUTPUT_MODEL_PATH}")

if __name__ == "__main__":
    main()