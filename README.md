# Champion–Challenger Model Comparison with Hugging Face and MLflow

## 1. Project Overview

This project implements a Champion–Challenger approach to compare two pretrained Hugging Face sentiment classification models using MLflow.

The objective is to evaluate both models on an identical dataset, compare their performance and inference latency, and select the most suitable model as the Champion.

The project also demonstrates MLflow experiment tracking, model registration, model versioning, alias assignment, and prediction on new text samples.

## 2. Models Selected

Two pretrained Hugging Face models are compared.

| Model | Hugging Face Model ID | Role |
|---|---|---|
| DistilBERT | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` | Initial Champion |
| Siebert-RoBERTa | `siebert/sentiment-roberta-large-english` | Challenger |

DistilBERT is initially selected as the Champion. Siebert-RoBERTa is evaluated as the Challenger to determine whether it can replace the existing Champion.

## 3. Dataset

**Original Dataset:** `news_sentiment_analysis.csv`

**Evaluation Dataset:** `data/eval_inputs.csv`

The original dataset contains news articles with sentiment labels.

The evaluation dataset is a fixed, balanced collection of 200 English-language samples.

| Dataset Property | Value |
|---|---:|
| Total evaluation samples | 200 |
| Positive samples | 100 |
| Negative samples | 100 |
| Language | English |
| Random seed | 42 |

Both models are evaluated using exactly the same input samples to ensure a fair comparison.

Neutral samples are excluded because the selected pretrained models perform binary sentiment classification.

The evaluation dataset is kept unchanged between model runs.

## 4. Champion–Challenger Selection Rule

Macro-F1 is used as the primary evaluation metric.

The initial Champion is DistilBERT, while Siebert-RoBERTa is the Challenger.

The Challenger can replace the Champion when:

1. Its Macro-F1 score improves by at least 0.01 absolute.
2. Its average inference latency does not exceed five times the Champion's average latency.

Additional selection conditions:

- If the absolute Macro-F1 difference is less than 0.01, the faster model is preferred.
- If the Challenger performs more than 0.01 worse, the existing Champion is retained.
- If the Challenger achieves the required F1 improvement but exceeds the latency limit, the existing Champion is retained.

This selection rule was defined before the final model evaluation.

## 5. Evaluation Metrics

The following metrics are recorded for both models:

- Accuracy
- Macro Precision
- Macro Recall
- Macro-F1 Score
- Average Inference Latency (milliseconds per sample)
- Total Inference Time (seconds)

Macro-F1 is the primary performance metric because it considers both positive and negative classes.

## 6. MLflow Experiment Tracking

MLflow is used to track and compare the experiments.

The following information is logged:

- Model parameters and configuration
- Evaluation metrics
- Model artifacts
- Prediction outputs
- Confusion matrices
- Classification reports
- Model comparison results
- Inference latency

**MLflow Experiment Name:** `Champion-Challenger-Sentiment`

**MLflow Tracking URI:** `http://127.0.0.1:5000`

## 7. Experimental Results

Both models were evaluated using the same 200 English-language news samples.

| Metric | DistilBERT | Siebert-RoBERTa |
|---|---:|---:|
| Accuracy | 0.5950 | 0.6300 |
| Macro Precision | 0.5951 | 0.6578 |
| Macro Recall | 0.5950 | 0.6300 |
| Macro-F1 | 0.5949 | 0.6129 |
| Average Latency (ms/sample) | 129.6276 | 900.7777 |
| Total Inference Time (seconds) | 25.9255 | 180.1555 |

### Champion Selection Result

**Selected Champion: DistilBERT**

Siebert-RoBERTa achieved a Macro-F1 score of 0.6129, compared with 0.5949 for DistilBERT.

Although the Challenger improved Macro-F1 by approximately 0.018, its average inference latency was approximately 6.95 times higher than DistilBERT.

The predefined selection rule limits Challenger latency to five times the Champion's latency.

Therefore, Siebert-RoBERTa did not qualify for promotion, and DistilBERT was retained as the Champion.

## 8. MLflow Model Registry

Both evaluated models were registered under the same MLflow registered model name:

`SentimentChampionChallenger`

| Model | Version | Alias |
|---|---:|---|
| DistilBERT | 1 | champion |
| Siebert-RoBERTa | 2 | challenger |

The `champion` alias points to DistilBERT, and the `challenger` alias points to Siebert-RoBERTa.

The model aliases allow the selected model versions to be accessed without hardcoding individual model version numbers.

## 9. Prediction on New Data

The registered Champion model is loaded using its MLflow alias:

`models:/SentimentChampionChallenger@champion`

The `predict_from_mlflow.py` script performs sentiment prediction on eight new text samples.

It generates the following information:

- Input text
- Predicted sentiment
- Prediction confidence

The predictions are saved to:

`outputs/new_data_champion_predictions.csv`

A separate MLflow run records the new-data prediction experiment.

## 10. Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Programming language |
| Hugging Face Transformers | Pretrained sentiment models |
| PyTorch | Model inference |
| MLflow | Experiment tracking and model registry |
| Pandas | Dataset processing |
| NumPy | Numerical operations |
| Scikit-learn | Evaluation metrics |
| Matplotlib | Result visualization |
| langdetect | Language detection |
| SQLAlchemy | MLflow backend database support |

## 11. Project Structure

```text
1MS25CI081-mlflow-champion-challenger/
│
├── data/
│   └── eval_inputs.csv
│
├── outputs/
│   ├── disagreement_examples.csv
│   ├── distilbert_classification_report.json
│   ├── distilbert_confusion_matrix.png
│   ├── distilbert_predictions.csv
│   ├── latency_comparison.png
│   ├── macro_f1_comparison.png
│   ├── metrics_comparison.json
│   ├── model_comparison.csv
│   ├── model_runs.json
│   ├── new_data_champion_predictions.csv
│   ├── roberta_classification_report.json
│   ├── roberta_confusion_matrix.png
│   └── roberta_predictions.csv
│
├── screenshots/
│   ├── 01_mlflow_experiments.png
│   ├── 02_model_registry.png
│   └── 03_new_predictions.png
│
├── evaluate.py
├── register_models.py
├── predict_from_mlflow.py
├── news_sentiment_analysis.csv
├── requirements.txt
├── README.md
└── .gitignore
```

Generated MLflow databases, artifacts, virtual environments, and IDE-specific files are excluded from version control.

## 12. Installation and Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/Mohammed-Zubiyaan-A/1MS25CI081-mlflow-champion-challenger.git
```

### Step 2: Navigate to the project directory

```bash
cd 1MS25CI081-mlflow-champion-challenger
```

### Step 3: Create a virtual environment

```bash
python -m venv venv
```

### Step 4: Activate the virtual environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 5: Install the dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 13. Running the Project

### Step 1: Start the MLflow Tracking Server

Open a terminal inside the project directory and run:

```bash
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000
```

Keep this terminal running.

### Step 2: Open the MLflow UI

Open the following address in your browser:

http://127.0.0.1:5000

### Step 3: Evaluate Both Models

Open a second terminal, activate the virtual environment, and run:

```bash
python evaluate.py
```

This evaluates DistilBERT and Siebert-RoBERTa on the fixed evaluation dataset and logs the results to MLflow.

### Step 4: Register the Models

```bash
python register_models.py
```

This registers both models and assigns the `champion` and `challenger` aliases.

### Step 5: Predict Sentiment on New Data

```bash
python predict_from_mlflow.py
```

This loads the registered Champion model and predicts sentiment for new text samples.

The prediction results are saved in the `outputs/` directory.

## 14. Project Outcomes

The project successfully demonstrates:

1. Comparison of two pretrained Hugging Face sentiment classification models.
2. Evaluation using a common fixed dataset.
3. MLflow experiment tracking and metric logging.
4. Champion–Challenger model selection using performance and latency.
5. Model registration and alias assignment.
6. Loading a registered model using its Champion alias.
7. Sentiment prediction on new text samples.
8. Storage of experimental results and visualizations.

## 15. Conclusion

This project demonstrates how the Champion–Challenger approach can be implemented using Hugging Face Transformers and MLflow.

Both models were evaluated using identical inputs and compared based on Macro-F1 and inference latency.

Although Siebert-RoBERTa achieved a higher Macro-F1 score, DistilBERT was retained as the Champion because the Challenger exceeded the predefined latency limit.

The project also demonstrates the practical use of MLflow for experiment tracking, model registry management, model versioning, alias-based model loading, and prediction on new data.
