import mlflow
import mlflow.transformers
import pandas as pd

from pathlib import Path
from mlflow.tracking import MlflowClient


# --------------------------------------------------
# Step 1: Configure MLflow
# --------------------------------------------------

TRACKING_URI = "http://127.0.0.1:5000"

REGISTERED_MODEL_NAME = "SentimentChampionChallenger"
MODEL_ALIAS = "champion"

MODEL_URI = (
    f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
)

mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient(tracking_uri=TRACKING_URI)


# --------------------------------------------------
# Step 2: Load new input data
# --------------------------------------------------

new_data_path = Path("data/new_data.csv")

if not new_data_path.exists():
    raise FileNotFoundError(
        "data/new_data.csv not found. Create the file first."
    )

new_data = pd.read_csv(new_data_path)

required_columns = {"sample_id", "text"}

if not required_columns.issubset(new_data.columns):
    raise ValueError(
        "new_data.csv must contain sample_id and text columns."
    )

if not 5 <= len(new_data) <= 10:
    raise ValueError(
        "The assignment requires between 5 and 10 new examples."
    )

if new_data["text"].isna().any():
    raise ValueError("New input data contains missing text.")

print("New prediction inputs loaded.")
print("Number of new samples:", len(new_data))


# --------------------------------------------------
# Step 3: Verify inputs are not in evaluation dataset
# --------------------------------------------------

eval_path = Path("data/eval_inputs.csv")

if not eval_path.exists():
    raise FileNotFoundError(
        "Evaluation dataset not found."
    )

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
print("No overlap with evaluation dataset.")


# --------------------------------------------------
# Step 4: Retrieve Champion version and model metadata
# --------------------------------------------------

print("\nRetrieving Champion model information...")

model_version = client.get_model_version_by_alias(
    REGISTERED_MODEL_NAME,
    MODEL_ALIAS
)

registered_version = str(model_version.version)

model_run = client.get_run(model_version.run_id)

hf_model_id = model_run.data.params.get(
    "hf_model_id",
    "Not available"
)

print("Registered model:", REGISTERED_MODEL_NAME)
print("Model alias:", MODEL_ALIAS)
print("Loaded model version:", registered_version)
print("Hugging Face model ID:", hf_model_id)


# --------------------------------------------------
# Step 5: Load Champion model through MLflow
# --------------------------------------------------

print("\nLoading Champion model from MLflow...")

classifier = mlflow.transformers.load_model(MODEL_URI)

print("Champion model loaded successfully!")


# --------------------------------------------------
# Step 6: Perform sentiment predictions
# --------------------------------------------------

print("\nRunning sentiment predictions...")

predictions = classifier(
    new_data["text"].tolist(),
    batch_size=8,
    truncation=True,
    max_length=256
)


# --------------------------------------------------
# Step 7: Prepare prediction results
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
print("=" * 100)

print(
    new_data[
        [
            "sample_id",
            "text",
            "predicted_sentiment",
            "confidence"
        ]
    ].to_string(index=False)
)


# --------------------------------------------------
# Step 8: Save predictions
# --------------------------------------------------

output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "new_data_predictions.csv"

new_data.to_csv(output_path, index=False)

print("\nPredictions saved to:", output_path)


# --------------------------------------------------
# Step 9: Log prediction activity in MLflow
# --------------------------------------------------

experiment = mlflow.get_experiment_by_name(
    "Champion-Challenger-Sentiment"
)

if experiment is None:
    raise ValueError(
        "MLflow experiment 'Champion-Challenger-Sentiment' not found."
    )

with mlflow.start_run(
    experiment_id=experiment.experiment_id,
    run_name="Champion_Prediction_New_Data"
):

    mlflow.log_param("model_uri", MODEL_URI)
    mlflow.log_param("model_alias", MODEL_ALIAS)
    mlflow.log_param("model_version", registered_version)
    mlflow.log_param("hf_model_id", hf_model_id)
    mlflow.log_param("number_of_samples", len(new_data))

    mlflow.set_tag("model_name", "DistilBERT")
    mlflow.set_tag("prediction_type", "new_unseen_data")
    mlflow.set_tag("registered_model_name", REGISTERED_MODEL_NAME)

    mlflow.log_artifact(str(new_data_path))
    mlflow.log_artifact(str(output_path))

    print("\nPrediction activity logged to MLflow.")


# --------------------------------------------------
# Step 10: Completion
# --------------------------------------------------

print("\n" + "=" * 60)
print("NEW DATA PREDICTION COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Champion model version:", registered_version)
print("Hugging Face model:", hf_model_id)
print("Output file:", output_path)
