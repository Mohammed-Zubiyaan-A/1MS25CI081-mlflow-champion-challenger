
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
# Step 2: Read existing model run information
# --------------------------------------------------

json_path = Path("outputs/model_runs.json")

with open(json_path, "r", encoding="utf-8") as file:
    run_data = json.load(file)

if not json_path.exists():
    raise FileNotFoundError("outputs/model_runs.json was not found.")

models = [
    {
        "name": run_data["model_a"]["name"],
        "model_uri": run_data["model_a"]["model_uri"],
        "run_id": run_data["model_a"]["run_id"],
        "alias": "champion",
    },
    {
        "name": run_data["model_b"]["name"],
        "model_uri": run_data["model_b"]["model_uri"],
        "run_id": run_data["model_b"]["run_id"],
        "alias": "challenger",
    },
]


# --------------------------------------------------
# Step 3: Create registered model if necessary
# --------------------------------------------------

try:
    client.get_registered_model(REGISTERED_MODEL_NAME)
    print(f"Registered model already exists: {REGISTERED_MODEL_NAME}")

except MlflowException:
    client.create_registered_model(REGISTERED_MODEL_NAME)
    print(f"Created registered model: {REGISTERED_MODEL_NAME}")


# --------------------------------------------------
# Step 4: Register both model versions
# --------------------------------------------------

for model in models:

    print("\n" + "=" * 55)
    print(f"Registering: {model['name']}")
    print("=" * 55)

    # Avoid creating duplicate versions if this script is rerun.
    existing_versions = client.search_model_versions(
        f"name='{REGISTERED_MODEL_NAME}'"
    )

    matching_version = next(
        (
            version for version in existing_versions
            if version.run_id == model["run_id"]
            and version.source == model["model_uri"]
        ),
        None,
    )

    if matching_version:
        model_version = matching_version
        print(f"Existing model version found: {model_version.version}")

    else:
        model_version = client.create_model_version(
            name=REGISTERED_MODEL_NAME,
            source=model["model_uri"],
            run_id=model["run_id"],
        )

        print(f"Created model version: {model_version.version}")

    # Wait until MLflow finishes registering the model.
    deadline = time.time() + 300

    while True:
        current_version = client.get_model_version(
            name=REGISTERED_MODEL_NAME,
            version=model_version.version,
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

    # Assign the requested alias.
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=model["alias"],
        version=model_version.version,
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
        alias=alias,
    )

    print(f"\nAlias: {alias}")
    print(f"Version: {registered_version.version}")
    print(f"Run ID: {registered_version.run_id}")
    print(f"Source: {registered_version.source}")

print("\nBoth models registered successfully!")
