
import gc
import json
import time
from pathlib import Path

import mlflow
import mlflow.transformers
import matplotlib.pyplot as plt
import pandas as pd
import torch

from transformers import pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ==========================================================
# 1. CONFIGURATION
# ==========================================================

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Champion-Challenger-Sentiment"

DATA_PATH = Path("data/eval_inputs.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_CONFIGS = [
    {
        "name": "DistilBERT",
        "model_id": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "role": "initial_champion",
        "filename": "distilbert",
    },
    {
        "name": "Siebert-RoBERTa",
        "model_id": "siebert/sentiment-roberta-large-english",
        "role": "challenger",
        "filename": "roberta",
    },
]

BATCH_SIZE = 8
MAX_LENGTH = 256
RANDOM_SEED = 42

LABELS = ["negative", "positive"]

# ==========================================================
# 2. MLflow CONFIGURATION
# ==========================================================

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

print("MLflow Tracking URI:", mlflow.get_tracking_uri())
print("Experiment:", EXPERIMENT_NAME)

# ==========================================================
# 3. LOAD EVALUATION DATASET
# ==========================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

required_columns = {
    "sample_id",
    "text",
    "actual_sentiment",
}

if not required_columns.issubset(df.columns):
    raise ValueError("Evaluation dataset has incorrect columns.")

if len(df) != 200:
    raise ValueError("Expected exactly 200 evaluation samples.")

if set(df["actual_sentiment"].unique()) != set(LABELS):
    raise ValueError("Expected positive and negative labels only.")

if df["text"].isna().any() or (df["text"].astype(str).str.strip() == "").any():
    raise ValueError("Evaluation dataset contains empty text.")

texts = df["text"].astype(str).tolist()
actual = df["actual_sentiment"].tolist()

print("\nEvaluation dataset loaded.")
print("Total samples:", len(df))
print(df["actual_sentiment"].value_counts())

# ==========================================================
# 4. NORMALIZE MODEL PREDICTIONS
# ==========================================================

def normalize_prediction(result, model):
    # Handle a possible nested pipeline output
    if isinstance(result, list):
        result = result[0]

    label = str(result["label"]).lower()

    # Handle generic LABEL_0 / LABEL_1 outputs
    if label.startswith("label_"):
        index = int(label.split("_")[-1])
        label = str(model.config.id2label[index]).lower()

    if "positive" in label:
        return "positive"

    if "negative" in label:
        return "negative"

    raise ValueError(
        f"Unexpected sentiment label: {label}"
    )

# ==========================================================
# 5. EVALUATE ONE MODEL
# ==========================================================

def evaluate_model(config):

    model_name = config["name"]
    model_id = config["model_id"]
    filename = config["filename"]

    print("\n" + "=" * 60)
    print("Loading:", model_name)
    print("Hugging Face ID:", model_id)
    print("=" * 60)

    with mlflow.start_run(run_name=f"{model_name}_Evaluation"):

        mlflow.log_params({
            "model_name": model_name,
            "hf_model_id": model_id,
            "task": "sentiment-classification",
            "dataset_size": len(df),
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "random_seed": RANDOM_SEED,
            "device": "CPU",
        })

        mlflow.set_tags({
            "model_role": config["role"],
            "hf_model_id": model_id,
            "evaluation_dataset": str(DATA_PATH),
        })

        # Load model on CPU
        classifier = pipeline(
            task="text-classification",
            model=model_id,
            tokenizer=model_id,
            device=-1,
        )

        classifier.model.eval()

        # Warm-up: excluded from inference timing
        classifier(
            ["This is a warm-up sentence."],
            truncation=True,
            max_length=MAX_LENGTH,
        )

        print("Running inference on", len(texts), "samples...")

        # Measure inference time
        start_time = time.perf_counter()

        raw_predictions = classifier(
            texts,
            batch_size=BATCH_SIZE,
            truncation=True,
            max_length=MAX_LENGTH,
        )

        total_time = time.perf_counter() - start_time

        predictions = [
            normalize_prediction(result, classifier.model)
            for result in raw_predictions
        ]

        average_latency_ms = (
            total_time / len(texts)
        ) * 1000

        # --------------------------------------------------
        # Calculate evaluation metrics
        # --------------------------------------------------

        accuracy = accuracy_score(actual, predictions)

        precision, recall, macro_f1, _ = (
            precision_recall_fscore_support(
                actual,
                predictions,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        )

        metrics = {
            "accuracy": float(accuracy),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(macro_f1),
            "average_latency_ms_per_sample": float(
                average_latency_ms
            ),
            "total_inference_time_seconds": float(
                total_time
            ),
        }

        print("\nEvaluation results:", model_name)

        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        # --------------------------------------------------
        # Save predictions
        # --------------------------------------------------

        prediction_df = df.copy()
        prediction_df["predicted_sentiment"] = predictions
        prediction_df["correct"] = (
            prediction_df["actual_sentiment"]
            == prediction_df["predicted_sentiment"]
        )

        prediction_path = (
            OUTPUT_DIR / f"{filename}_predictions.csv"
        )

        prediction_df.to_csv(
            prediction_path,
            index=False,
        )

        # --------------------------------------------------
        # Classification report
        # --------------------------------------------------

        report = classification_report(
            actual,
            predictions,
            labels=LABELS,
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        )

        report_path = (
            OUTPUT_DIR / f"{filename}_classification_report.json"
        )

        with open(report_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

        # --------------------------------------------------
        # Confusion matrix
        # --------------------------------------------------

        cm = confusion_matrix(
            actual,
            predictions,
            labels=LABELS,
        )

        fig, ax = plt.subplots(figsize=(6, 5))

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=LABELS,
        )

        display.plot(ax=ax)

        ax.set_title(f"{model_name} - Confusion Matrix")

        plt.tight_layout()

        confusion_path = (
            OUTPUT_DIR / f"{filename}_confusion_matrix.png"
        )

        plt.savefig(confusion_path, dpi=150)
        plt.close(fig)

        # --------------------------------------------------
        # Log metrics and artifacts to MLflow
        # --------------------------------------------------

        mlflow.log_metrics(metrics)

        mlflow.log_artifact(str(prediction_path))
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(confusion_path))

        # Log the Hugging Face model into MLflow
        print("\nLogging model artifact to MLflow...")

        mlflow.transformers.log_model(
            transformers_model=classifier,
            artifact_path="model",
            task="text-classification",
            metadata={"hf_model_id": model_id},
        )

        run_id = mlflow.active_run().info.run_id

        print("\nMLflow Run ID:", run_id)
        print("Model logged successfully.")

    # Release memory before loading the next model
    del classifier
    del raw_predictions

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "name": model_name,
        "model_id": model_id,
        "run_id": run_id,
        "model_uri": f"runs:/{run_id}/model",
        "metrics": metrics,
        "prediction_path": str(prediction_path),
    }


# ==========================================================
# 6. RUN BOTH MODELS SEQUENTIALLY
# ==========================================================

results = []

for config in MODEL_CONFIGS:
    result = evaluate_model(config)
    results.append(result)

# ==========================================================
# 7. COMPARE MODELS
# ==========================================================

model_a = results[0]
model_b = results[1]

f1_a = model_a["metrics"]["macro_f1"]
f1_b = model_b["metrics"]["macro_f1"]

latency_a = model_a["metrics"]["average_latency_ms_per_sample"]
latency_b = model_b["metrics"]["average_latency_ms_per_sample"]

f1_difference = f1_b - f1_a

latency_ratio = (
    latency_b / latency_a if latency_a > 0 else float("inf")
)

# Champion selection rule
if f1_difference >= 0.01 and latency_ratio <= 5:
    selected_champion = model_b["name"]
    decision = "Challenger replaces initial Champion."

elif -0.01 < f1_difference < 0.01:
    if latency_b < latency_a:
        selected_champion = model_b["name"]
        decision = "Challenger selected because it is faster."
    else:
        selected_champion = model_a["name"]
        decision = "Initial Champion retained because it is faster."

else:
    selected_champion = model_a["name"]
    decision = "Initial Champion retained."

print("\n" + "=" * 60)
print("CHAMPION - CHALLENGER COMPARISON")
print("=" * 60)

print("Model A Macro-F1:", round(f1_a, 4))
print("Model B Macro-F1:", round(f1_b, 4))
print("F1 improvement:", round(f1_difference, 4))
print("Latency ratio:", round(latency_ratio, 2))
print("Selected Champion:", selected_champion)
print("Decision:", decision)

# ==========================================================
# 8. SAVE COMPARISON RESULTS
# ==========================================================

comparison = pd.DataFrame([
    {
        "model": result["name"],
        **result["metrics"],
    }
    for result in results
])

comparison_path = OUTPUT_DIR / "model_comparison.csv"
comparison.to_csv(comparison_path, index=False)

metrics_json = {
    result["name"]: result["metrics"]
    for result in results
}

metrics_path = OUTPUT_DIR / "metrics_comparison.json"

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(metrics_json, file, indent=4)

# ==========================================================
# 9. CREATE COMPARISON PLOTS
# ==========================================================

# Macro-F1 comparison
fig, ax = plt.subplots(figsize=(7, 5))

ax.bar(
    comparison["model"],
    comparison["macro_f1"],
)

ax.set_title("Macro-F1 Comparison")
ax.set_ylabel("Macro-F1 Score")
ax.set_ylim(0, 1)

plt.tight_layout()

f1_plot = OUTPUT_DIR / "macro_f1_comparison.png"
plt.savefig(f1_plot, dpi=150)
plt.close(fig)

# Average latency comparison
fig, ax = plt.subplots(figsize=(7, 5))

ax.bar(
    comparison["model"],
    comparison["average_latency_ms_per_sample"],
)

ax.set_title("Average Inference Latency Comparison")
ax.set_ylabel("Milliseconds per sample")

plt.tight_layout()

latency_plot = OUTPUT_DIR / "latency_comparison.png"
plt.savefig(latency_plot, dpi=150)
plt.close(fig)

# ==========================================================
# 10. FIND DISAGREEMENT EXAMPLES
# ==========================================================

pred_a = pd.read_csv(model_a["prediction_path"])
pred_b = pd.read_csv(model_b["prediction_path"])

merged = pred_a[
    ["sample_id", "text", "actual_sentiment", "predicted_sentiment"]
].merge(
    pred_b[["sample_id", "predicted_sentiment"]],
    on="sample_id",
    suffixes=("_model_a", "_model_b"),
)

disagreements = merged[
    merged["predicted_sentiment_model_a"]
    != merged["predicted_sentiment_model_b"]
].head(5)

disagreement_path = OUTPUT_DIR / "disagreement_examples.csv"
disagreements.to_csv(disagreement_path, index=False)

# ==========================================================
# 11. SUMMARY RUN IN MLFLOW
# ==========================================================

with mlflow.start_run(run_name="Champion_Challenger_Summary"):

    mlflow.log_metrics({
        "model_a_macro_f1": f1_a,
        "model_b_macro_f1": f1_b,
        "f1_difference": f1_difference,
        "latency_ratio": latency_ratio,
    })

    mlflow.set_tags({
        "selected_champion": selected_champion,
        "decision": decision,
        "comparison_metric": "macro_f1",
    })

    for result in results:
        mlflow.log_metric(
            f"{result['name']}_accuracy",
            result["metrics"]["accuracy"],
        )

    for artifact in [
        comparison_path,
        metrics_path,
        f1_plot,
        latency_plot,
        disagreement_path,
    ]:
        mlflow.log_artifact(str(artifact))

    summary_run_id = mlflow.active_run().info.run_id

# ==========================================================
# 12. SAVE RUN INFORMATION FOR REGISTRATION
# ==========================================================

run_information = {
    "experiment_name": EXPERIMENT_NAME,
    "model_a": model_a,
    "model_b": model_b,
    "selected_champion": selected_champion,
    "decision": decision,
    "summary_run_id": summary_run_id,
}

run_info_path = OUTPUT_DIR / "model_runs.json"

with open(run_info_path, "w", encoding="utf-8") as file:
    json.dump(run_information, file, indent=4)

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Selected Champion:", selected_champion)
print("Summary Run ID:", summary_run_id)
print("Results saved in:", OUTPUT_DIR)
