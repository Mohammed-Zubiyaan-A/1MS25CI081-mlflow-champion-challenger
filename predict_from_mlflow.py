import os
import mlflow
import pandas as pd

from mlflow.tracking import MlflowClient


# ============================================================
# STEP 1: CONFIGURATION
# ============================================================

mlflow.set_tracking_uri("http://127.0.0.1:5000")

MODEL_NAME = "SentimentChampionChallenger"
MODEL_URI = f"models:/{MODEL_NAME}@champion"

NEW_DATA_PATH = "data/new_data.csv"
EVAL_DATA_PATH = "data/eval_inputs.csv"
OUTPUT_PATH = "outputs/new_data_predictions.csv"


# ============================================================
# STEP 2: CHECK INPUT FILES
# ============================================================

if not os.path.exists(NEW_DATA_PATH):
    raise FileNotFoundError(
        f"New data file not found: {NEW_DATA_PATH}"
    )

if not os.path.exists(EVAL_DATA_PATH):
    raise FileNotFoundError(
        f"Evaluation data file not found: {EVAL_DATA_PATH}"
    )

new_data = pd.read_csv(NEW_DATA_PATH)
eval_data = pd.read_csv(EVAL_DATA_PATH)

if "text" not in new_data.columns:
    raise ValueError(
        "The new_data.csv file must contain a 'text' column."
    )

if not 5 <= len(new_data) <= 10:
    raise ValueError(
        "Provide between 5 and 10 new examples."
    )

new_data = new_data.dropna(subset=["text"]).copy()
new_data["text"] = new_data["text"].astype(str).str.strip()

if new_data["text"].eq("").any():
    raise ValueError("New data contains empty text examples.")

if not 5 <= len(new_data) <= 10:
    raise ValueError(
        "After removing empty rows, provide 5 to 10 examples."
    )


# ============================================================
# STEP 3: ENSURE NEW DATA IS NOT IN EVALUATION DATA
# ============================================================

eval_texts = set(
    eval_data["text"].dropna().astype(str).str.strip()
)

new_texts = set(new_data["text"])

overlapping_texts = new_texts.intersection(eval_texts)

if overlapping_texts:
    raise ValueError(
        "Some new examples already exist in the evaluation dataset. "
        "Use completely new examples."
    )

print("New data validation successful!")
print("Number of new examples:", len(new_data))


# ============================================================
# STEP 4: LOAD THE CHAMPION MODEL FROM MLFLOW
# ============================================================

client = MlflowClient()

# Retrieve the model version assigned the champion alias
model_version = client.get_model_version_by_alias(
    MODEL_NAME,
    "champion"
)

# Retrieve the original evaluation run
source_run = client.get_run(model_version.run_id)

# Read model information from the original run parameters
model_name = source_run.data.params.get(
    "model_name", "Unknown"
)

hf_model_id = source_run.data.params.get(
    "hf_model_id", "Unknown"
)

print("\n========== CHAMPION MODEL DETAILS ==========")
print("Registered Model:", MODEL_NAME)
print("Loaded Model Version:", model_version.version)
print("Model Name:", model_name)
print("Hugging Face Model ID:", hf_model_id)
print("Source Run ID:", model_version.run_id)

# Load the Champion model using MLflow PyFunc
model = mlflow.pyfunc.load_model(MODEL_URI)


# ============================================================
# STEP 5: GENERATE PREDICTIONS USING MLFLOW
# ============================================================

print("\nGenerating predictions...")

# Use the MLflow PyFunc model for prediction
preds = model.predict(new_data["text"])


# ============================================================
# STEP 6: PROCESS PREDICTION OUTPUT
# ============================================================

def extract_prediction(pred):
    """
    Extract the predicted label and confidence score
    from the MLflow PyFunc prediction output.
    """

    # Handle pandas Series
    if isinstance(pred, pd.Series):
        pred = pred.to_dict()

    # Handle pandas DataFrame row converted to dictionary
    if isinstance(pred, dict):

        # Handle nested predictions
        if "predictions" in pred:
            return extract_prediction(pred["predictions"])

        label = pred.get("label", "Unknown")
        score = pred.get("score", None)

        if score is not None:
            score = float(score)

        return label, score

    # Handle nested list output from Transformers
    if isinstance(pred, list):
        if len(pred) == 0:
            return "Unknown", None

        return extract_prediction(pred[0])

    # Handle string-only output
    if isinstance(pred, str):
        return pred, None

    return str(pred), None


predicted_labels = []
confidence_scores = []

# Handle DataFrame output
if isinstance(preds, pd.DataFrame):

    for _, row in preds.iterrows():
        label, score = extract_prediction(row.to_dict())

        predicted_labels.append(label)
        confidence_scores.append(score)

# Handle Series output
elif isinstance(preds, pd.Series):

    for pred in preds.tolist():
        label, score = extract_prediction(pred)

        predicted_labels.append(label)
        confidence_scores.append(score)

# Handle list or array-like output
else:

    for pred in preds:
        label, score = extract_prediction(pred)

        predicted_labels.append(label)
        confidence_scores.append(score)


if len(predicted_labels) != len(new_data):
    raise ValueError(
        "Prediction count does not match the number of input examples."
    )

new_data["predicted_sentiment"] = predicted_labels
new_data["confidence"] = confidence_scores


# ============================================================
# STEP 7: DISPLAY PREDICTION RESULTS
# ============================================================

print("\n========== NEW DATA PREDICTIONS ==========")

display_columns = ["text", "predicted_sentiment", "confidence"]

print(
    new_data[display_columns].to_string(index=False)
)


# ============================================================
# STEP 8: SAVE PREDICTIONS TO CSV
# ============================================================

os.makedirs("outputs", exist_ok=True)

new_data.to_csv(OUTPUT_PATH, index=False)

print("\nPredictions saved successfully!")
print("Output file:", OUTPUT_PATH)


# ============================================================
# STEP 9: LOG PREDICTION RUN IN MLFLOW
# ============================================================

with mlflow.start_run(
    experiment_id=source_run.info.experiment_id,
    run_name="Champion_New_Data_Predictions"
):

    mlflow.set_tag("model_name", model_name)
    mlflow.set_tag("hf_model_id", hf_model_id)
    mlflow.set_tag("model_role", "champion")
    mlflow.set_tag("model_version", str(model_version.version))

    mlflow.log_param("model_uri", MODEL_URI)
    mlflow.log_param("prediction_samples", len(new_data))

    mlflow.log_artifact(OUTPUT_PATH)

    mlflow.log_metric(
        "prediction_samples",
        len(new_data)
    )

    print("\nPrediction run logged successfully in MLflow!")


# ============================================================
# STEP 10: FINAL STATUS
# ============================================================

print("\n========== EXECUTION COMPLETED ==========")
print("Champion Model:", model_name)
print("Model Version:", model_version.version)
print("Total Predictions:", len(new_data))
print("Saved Predictions:", OUTPUT_PATH)
