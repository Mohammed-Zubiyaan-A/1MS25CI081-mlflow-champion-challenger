
import json
import time
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException


# --------------------------------------------------
# Step 1: Configure MLflow
# --------------------------------------------------

TRACKING_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "SentimentChampionChallenger"

mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient(tracking_uri=TRACKING_URI)


# --------------------------------------------------
# Step 2: Read evaluation results
# --------------------------------------------------

json_path = Path("outputs/model_runs.json")

if not json_path.exists():
    raise FileNotFoundError(
        "outputs/model_runs.json was not found."
    )

with open(json_path, "r", encoding="utf-8") as file:
    run_data = json.load(file)


# Read the winner selected by the evaluation script.
selected_champion = run_data["selected_champion"]

model_a = run_data["model_a"]
model_b = run_data["model_b"]

available_models = {
    model_a["name"]: model_a,
    model_b["name"]: model_b
}

if selected_champion not in available_models:
    raise ValueError(
        f"Unknown Champion: {selected_champion}"
    )

# Assign aliases dynamically using the evaluation decision.
models = []

for model_name, model_data in available_models.items():

    alias = (
        "champion"
        if model_name == selected_champion
        else "challenger"
    )

    models.append({
        "name": model_name,
        "model_uri": model_data["model_uri"],
        "run_id": model_data["run_id"],
        "alias": alias
    })

print("Selected Champion:", selected_champion)
print("Registry:", REGISTERED_MODEL_NAME)


# --------------------------------------------------
# Step 3: Create registered model if necessary
# --------------------------------------------------

try:
    client.get_registered_model(REGISTERED_MODEL_NAME)

    print(
        f"\nRegistered model already exists: "
        f"{REGISTERED_MODEL_NAME}"
    )

except MlflowException:
    client.create_registered_model(REGISTERED_MODEL_NAME)

    print(
        f"\nCreated registered model: "
        f"{REGISTERED_MODEL_NAME}"
    )


# --------------------------------------------------
# Step 4: Register both model versions
# --------------------------------------------------

for model in models:

    print("\n" + "=" * 55)
    print(f"Registering: {model['name']}")
    print(f"Required alias: {model['alias']}")
    print("=" * 55)

    # Verify that the run ID and model URI match.
    expected_uri = f"runs:/{model['run_id']}/model"

    if model["model_uri"] != expected_uri:
        raise ValueError(
            f"Model URI does not match run ID for "
            f"{model['name']}."
        )

    # Avoid duplicate registration of the same run.
    existing_versions = client.search_model_versions(
        f"name='{REGISTERED_MODEL_NAME}'"
    )

    matching_version = next(
        (
            version for version in existing_versions
            if version.run_id == model["run_id"]
            and version.source == model["model_uri"]
        ),
        None
    )

    if matching_version:

        model_version = matching_version

        print(
            f"Existing model version found: "
            f"{model_version.version}"
        )

    else:

        model_version = client.create_model_version(
            name=REGISTERED_MODEL_NAME,
            source=model["model_uri"],
            run_id=model["run_id"]
        )

        print(
            f"Created model version: "
            f"{model_version.version}"
        )

    # Wait for model registration to finish.
    deadline = time.time() + 300

    while True:

        current_version = client.get_model_version(
            name=REGISTERED_MODEL_NAME,
            version=model_version.version
        )

        if current_version.status == "READY":
            break

        if current_version.status == "FAILED":
            raise RuntimeError(
                f"Registration failed for {model['name']}: "
                f"{current_version.status_message}"
            )

        if time.time() > deadline:
            raise TimeoutError(
                f"Registration timed out for {model['name']}."
            )

        time.sleep(2)

    # Assign the alias.
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=model["alias"],
        version=model_version.version
    )

    print(f"Model: {model['name']}")
    print(f"Version: {model_version.version}")
    print(f"Alias: {model['alias']}")
    print("Registration status: READY")


# --------------------------------------------------
# Step 5: Verify both aliases
# --------------------------------------------------

print("\n" + "=" * 55)
print("MODEL REGISTRY VERIFICATION")
print("=" * 55)

for alias in ["champion", "challenger"]:

    registered_version = client.get_model_version_by_alias(
        name=REGISTERED_MODEL_NAME,
        alias=alias
    )

    print(f"\nAlias: {alias}")
    print(f"Version: {registered_version.version}")
    print(f"Run ID: {registered_version.run_id}")
    print(f"Source: {registered_version.source}")

    # Verify that the assigned alias matches the evaluation.
    expected_model = next(
        model for model in models
        if model["alias"] == alias
    )

    if registered_version.run_id != expected_model["run_id"]:
        raise RuntimeError(
            f"Incorrect model assigned to alias: {alias}"
        )

print("\nBoth models registered successfully!")
print("Registry aliases match the evaluation decision.")
