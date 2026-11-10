
import mlflow
import mlflow.transformers
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# Step 1: Configure MLflow
# --------------------------------------------------

TRACKING_URI = "http://127.0.0.1:5000"

MODEL_URI = "models:/SentimentChampionChallenger@champion"

mlflow.set_tracking_uri(TRACKING_URI)


# --------------------------------------------------
# Step 2: Prepare new input sentences
# --------------------------------------------------

new_data = pd.DataFrame({
    "sample_id": [
        "new_001",
        "new_002",
        "new_003",
        "new_004",
        "new_005",
        "new_006",
        "new_007",
        "new_008"
    ],
    "text": [
        "The company reported excellent profits this quarter.",
        "The new product launch was a complete disappointment.",
        "Investors are optimistic about the company's future growth.",
        "The business suffered significant losses this year.",
        "The customer service team provided an outstanding experience.",
        "The latest announcement created serious concerns among investors.",
        "The company successfully expanded into international markets.",
        "The declining sales figures disappointed shareholders."
    ]
})


# --------------------------------------------------
# Step 3: Verify inputs are not in evaluation dataset
# --------------------------------------------------

eval_path = Path("data/eval_inputs.csv")

if not eval_path.exists():
    raise FileNotFoundError("Evaluation dataset not found.")

evaluation_data = pd.read_csv(eval_path)

evaluation_texts = set(
    evaluation_data["text"].astype(str).str.strip()
)

new_texts = set(
    new_data["text"].astype(str).str.strip()
)

if new_texts.intersection(evaluation_texts):
    raise ValueError(
        "Some prediction inputs already exist in the evaluation dataset."
    )

print("New prediction inputs verified.")
print("Number of new samples:", len(new_data))


# --------------------------------------------------
# Step 4: Load Champion from MLflow Model Registry
# --------------------------------------------------

print("\nLoading Champion model from MLflow...")

classifier = mlflow.transformers.load_model(MODEL_URI)

print("Champion model loaded successfully!")


# --------------------------------------------------
# Step 5: Perform sentiment prediction
# --------------------------------------------------

print("\nRunning sentiment predictions...")

predictions = classifier(
    new_data["text"].tolist(),
    batch_size=8,
    truncation=True,
    max_length=256
)


# --------------------------------------------------
# Step 6: Prepare prediction results
# --------------------------------------------------

new_data["predicted_sentiment"] = [
    prediction["label"].lower()
    for prediction in predictions
]

new_data["confidence"] = [
    round(float(prediction["score"]), 4)
    for prediction in predictions
]

print("\nPREDICTION RESULTS")
print("=" * 90)

print(
    new_data[
        ["sample_id", "text", "predicted_sentiment", "confidence"]
    ].to_string(index=False)
)


# --------------------------------------------------
# Step 7: Save predictions as CSV
# --------------------------------------------------

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "new_data_champion_predictions.csv"

new_data.to_csv(output_path, index=False)

print("\nPredictions saved to:", output_path)


# --------------------------------------------------
# Step 8: Log prediction activity in MLflow
# --------------------------------------------------

experiment = mlflow.get_experiment_by_name(
    "Champion-Challenger-Sentiment"
)

with mlflow.start_run(
    experiment_id=experiment.experiment_id,
    run_name="Champion_Prediction_New_Data"
):

    mlflow.log_param("model_uri", MODEL_URI)
    mlflow.log_param("model_alias", "champion")
    mlflow.log_param("number_of_samples", len(new_data))

    mlflow.set_tag("model_name", "DistilBERT")
    mlflow.set_tag("prediction_type", "new_unseen_data")

    mlflow.log_artifact(str(output_path))

    print("\nPrediction activity logged to MLflow.")

print("\n" + "=" * 60)
print("NEW DATA PREDICTION COMPLETED SUCCESSFULLY")
print("=" * 60)
