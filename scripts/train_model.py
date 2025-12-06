import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sklearn.model_selection import train_test_split

# ============================= PATH ============================= #
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# ============================= CONFIGURATION ============================= #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "artifacts", "golden_dataset_v2.csv")
OUTPUT_MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "my_fine_tuned_model")
CHART_PATH = os.path.join(BASE_DIR, "artifacts", "training_performance.png")

PRETRAINED_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5

# ============================= UTILITIES ============================= #
def extract_score(score_obj):
    """
    Extract float score from evaluator result.
    Supports both legacy (float) and new (dict) formats.

    Args:
        score_obj: Evaluator result (float or dict).

    Returns:
        float: Extracted score.
    """
    if isinstance(score_obj, dict):
        if "spearman" in score_obj:
            return score_obj["spearman"]
        return list(score_obj.values())[0]
    return score_obj

# ============================= TRAINING PIPELINE ============================= #
def main():
    """
    Fine-tune a CrossEncoder model on the golden dataset.
    Steps:
    1. Load dataset
    2. Split into train/validation
    3. Initialize model
    4. Train and evaluate per epoch
    5. Save final model and training chart
    """
    # --- Load dataset ---
    if not os.path.exists(DATASET_PATH):
        print("(X) Dataset file not found!")
        return

    print("Loading Golden Dataset V2...")
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=["score"])

    print(f"   -> Total samples: {len(df)}")
    print("   -> Score distribution:", dict(df["score"].value_counts()))

    # Train/validation split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_examples = [
        InputExample(texts=[r["jd_text"], r["cv_text"]], label=float(r["score"]))
        for _, r in train_df.iterrows()
    ]
    val_examples = [
        InputExample(texts=[r["jd_text"], r["cv_text"]], label=float(r["score"]))
        for _, r in val_df.iterrows()
    ]

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    evaluator = CECorrelationEvaluator.from_input_examples(val_examples, name="val_evaluator")

    # --- Initialize model ---
    print(f"(-) Loading pretrained model: {PRETRAINED_MODEL}...")
    model = CrossEncoder(PRETRAINED_MODEL, num_labels=1)

    # --- Training loop ---
    print("\n🚀 Starting training (epoch by epoch)...")

    history = {"epoch": [], "spearman_score": []}

    # Baseline evaluation
    print("Evaluating baseline model...")
    raw_score = evaluator(model)
    baseline_score = extract_score(raw_score)

    history["epoch"].append(0)
    history["spearman_score"].append(baseline_score)
    print(f"   Epoch 0 (Untrained): Spearman = {baseline_score:.4f}")

    # Epoch training
    for epoch in range(1, NUM_EPOCHS + 1):
        model.fit(
            train_dataloader=train_dataloader,
            epochs=1,
            warmup_steps=int(len(train_dataloader) * 0.1),
            optimizer_params={"lr": LEARNING_RATE},
            show_progress_bar=True,
        )

        raw_score = evaluator(model)
        current_score = extract_score(raw_score)

        history["epoch"].append(epoch)
        history["spearman_score"].append(current_score)

        trend = "↑ Improved" if current_score > baseline_score else "↓ Declined"
        print(f"   (V) Epoch {epoch}/{NUM_EPOCHS}: Spearman = {current_score:.4f} ({trend})")

    # --- Save model ---
    print("\n💾 Saving final model...")
    model.save(OUTPUT_MODEL_PATH)
    print(f"(V) Model saved at: {OUTPUT_MODEL_PATH}")

    # --- Plot training chart ---
    plt.figure(figsize=(10, 6))
    plt.plot(
        history["epoch"],
        history["spearman_score"],
        marker="o",
        linestyle="-",
        color="b",
        label="Validation Score",
    )
    plt.title("Model Learning Progress (Knowledge Distillation)", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Spearman Correlation (Higher is Better)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(history["epoch"])

    for i, txt in enumerate(history["spearman_score"]):
        plt.annotate(
            f"{txt:.3f}",
            (history["epoch"][i], history["spearman_score"][i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    plt.savefig(CHART_PATH)
    print(f"Training chart saved at: {CHART_PATH}")


if __name__ == "__main__":
    main()