# Champion–Challenger Model Comparison with Hugging Face and MLflow

## 1. Project Overview

This project implements a Champion–Challenger model comparison system using Hugging Face Transformers and MLflow. The objective is to evaluate two pretrained sentiment classification models on the same evaluation dataset, compare their performance and inference speed, and select the best model using a predefined decision rule.

The project uses MLflow Tracking to record experiments, parameters, metrics, artifacts, and model information. The selected Champion model is registered in the MLflow Model Registry and accessed through a model alias for predictions on new data.

**Selected Champion:** DistilBERT  
**Challenger:** Siebert-RoBERTa  
**Task:** Binary Sentiment Classification  
**Dataset:** News Sentiment Analysis Dataset

---

## 2. Objectives

- Compare two Hugging Face sentiment classification models using identical evaluation inputs.
- Track model parameters, evaluation metrics, latency, and artifacts using MLflow.
- Apply a predefined Champion selection rule based on Macro-F1 and inference latency.
- Register both models in the MLflow Model Registry with Champion and Challenger aliases.
- Load the selected Champion through MLflow and generate predictions on previously unseen data.
- Analyse model disagreements, performance differences, and limitations.

---

## 3. Models Selected

Two pretrained Hugging Face models were selected for binary sentiment classification.

| Model | Hugging Face Model ID | Role |
|---|---|---|
| DistilBERT | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` | Initial Champion |
| Siebert-RoBERTa | `siebert/sentiment-roberta-large-english` | Challenger |

Both models were evaluated using the same 200 samples from `data/eval_inputs.csv`.

### Model A – DistilBERT

DistilBERT is a smaller and faster version of BERT. It is designed to provide a balance between performance and computational efficiency.

### Model B – Siebert-RoBERTa

Siebert-RoBERTa is a sentiment classification model based on the RoBERTa architecture. It was selected as the Challenger to compare its classification performance and inference latency against DistilBERT.

---

## 4. Dataset Description

The original dataset used for this project is `news_sentiment_analysis.csv`.

It contains 3,500 news records with the following columns:

- Source
- Author
- Title
- Description
- URL
- Published At
- Sentiment
- Type

The `Description` column was used as the input text, and the `Sentiment` column was used as the reference label.

The original dataset contains positive, negative, and neutral sentiment labels. For this experiment, 200 English-language samples were selected and balanced between two classes.

| Sentiment | Number of Samples |
|---|---:|
| Positive | 100 |
| Negative | 100 |
| **Total** | **200** |

The selected samples were stored in `data/eval_inputs.csv`.

The evaluation dataset was fixed and shared between both models to ensure a fair comparison. Both models received identical input examples.

A random seed of 42 was used during sample selection.

---

## 5. Champion Selection Rule

The Champion selection rule was defined before evaluating the final model results.

### Primary Metric

**Macro-F1 Score**

Macro-F1 is used because it calculates the F1 score independently for each class and takes their average. A higher Macro-F1 score indicates better overall classification performance across the evaluated classes.

### Selection Criteria

1. The Challenger must improve Macro-F1 by at least **0.01 absolute** compared with the current Champion.
2. The Challenger's average inference latency must not exceed **5 times** the Champion's average inference latency.
3. If the absolute Macro-F1 difference is less than 0.01, the faster model is selected.
4. If the Challenger improves Macro-F1 by at least 0.01 but violates the latency limit, the existing Champion is retained.
5. If the Challenger performs worse by more than the permitted margin, the existing Champion is retained.

**Metric direction:** Higher Macro-F1 is better, while lower inference latency is better.

The selection rule was established before the final evaluation and was not changed after observing the results.

---

## 6. Evaluation Methodology

Both models were evaluated using the same evaluation dataset and experimental configuration.

### Experimental Configuration

| Parameter | Value |
|---|---|
| Task | Sentiment Classification |
| Evaluation Samples | 200 |
| Random Seed | 42 |
| Device | CPU |
| Batch Size | 8 |
| Maximum Sequence Length | 256 |
| Primary Metric | Macro-F1 |
| Tracking Tool | MLflow |
| Model Library | Hugging Face Transformers |

### Evaluation Metrics

The following metrics were recorded for both models:

- Accuracy
- Macro Precision
- Macro Recall
- Macro-F1 Score
- Average Inference Latency
- Total Inference Time

### MLflow Tracking

A single MLflow experiment was used to track the model evaluation runs.

The experiment contains:

1. `A_sentiment_distilbert-sst2` – DistilBERT evaluation run.
2. `B_sentiment_roberta-large` – Siebert-RoBERTa evaluation run.
3. `compare_sentiment` – Champion–Challenger comparison and selection run.

The model runs record their parameters, metrics, tags, predictions, classification reports, confusion matrices, and logged model artifacts.

The comparison run records the primary metric, selection margin, selected winner, and performance difference.

---

## 7. Evaluation Results

The following results were obtained from the final evaluation runs.

### Model Performance Comparison

| Metric | DistilBERT | Siebert-RoBERTa |
|---|---:|---:|
| Accuracy | 0.5950 | 0.6300 |
| Macro Precision | 0.5951 | 0.6578 |
| Macro Recall | 0.5950 | 0.6300 |
| Macro-F1 | 0.5949 | 0.6129 |
| Average Latency (ms/sample) | 137.3958 | 819.2594 |
| Total Inference Time (seconds) | 27.4792 | 163.8519 |

### Performance Difference

- Macro-F1 improvement: approximately **0.0180**.
- Average latency ratio: approximately **5.96×**.
- The Challenger achieved a higher Macro-F1 score.
- The Challenger exceeded the predefined 5× latency limit.

### Evaluation Plots

The project generates the following comparison visualizations:

1. Macro-F1 comparison between DistilBERT and Siebert-RoBERTa.
2. Average inference latency comparison between the two models.
3. Confusion matrices for both models.

These visualizations are saved in the `outputs/` directory.

---

## 8. Champion Selection Result

**Selected Champion: DistilBERT (Registered Model Version 5)**

Siebert-RoBERTa achieved a Macro-F1 score of 0.6129, compared with 0.5949 for DistilBERT. This represents an absolute improvement of approximately 0.0180, exceeding the predefined minimum improvement margin of 0.01. However, model selection also considers inference efficiency. DistilBERT recorded an average inference latency of 137.3958 milliseconds per sample, whereas Siebert-RoBERTa required 819.2594 milliseconds per sample. Therefore, the Challenger was approximately 5.96 times slower than the initial Champion.

According to the selection rule established before evaluation, a Challenger must improve Macro-F1 by at least 0.01 while keeping its average latency within five times the Champion's latency. Although Siebert-RoBERTa satisfied the performance improvement condition, it failed the latency guardrail. Consequently, DistilBERT was retained as the Champion.

The evaluation used 200 balanced English-language news samples, with 100 positive and 100 negative examples. This sample size provides an initial comparison but is not sufficient to establish general performance across all news domains. The pretrained models were not specifically fine-tuned on this dataset, which may contribute to classification errors. DistilBERT offers substantially faster CPU inference, while Siebert-RoBERTa provides a modest improvement in Macro-F1 at a considerable speed cost. Based on the predefined criteria and observed trade-off, DistilBERT is the selected Champion for this project.

---

## 9. MLflow Model Registry

Both evaluated models were registered under the same registered model name:

`SentimentChampionChallenger`

The model aliases were assigned according to the Champion selection decision.

| Model | Registered Model Version | Alias | Status |
|---|---:|---|---|
| DistilBERT | 5 | `champion` | READY |
| Siebert-RoBERTa | 6 | `challenger` | READY |

### Champion Model URI

```text
models:/SentimentChampionChallenger@champion
```

### Challenger Model URI

```text
models:/SentimentChampionChallenger@challenger
```

The aliases allow the selected model to be accessed without hard-coding a specific model version in the prediction script.

The registration process is implemented in `register_champion.py`.

---

## 10. Prediction on New Data

After selecting and registering the Champion model, predictions were generated on eight new news-related text examples.

The new examples were stored in:

```text
data/new_data.csv
```

These examples were not part of the 200-sample evaluation dataset.

The prediction script:

1. Loads the registered model using the Champion alias.
2. Retrieves the model version from the MLflow Model Registry.
3. Retrieves the Hugging Face model ID from the logged model run.
4. Loads the model through MLflow.
5. Generates predictions on the new input examples.
6. Displays the predicted sentiment and confidence scores.
7. Saves the results to `outputs/new_data_predictions.csv`.

The prediction script uses the MLflow model rather than directly loading or calling a Hugging Face pipeline.

### Prediction Model URI

```text
models:/SentimentChampionChallenger@champion
```

### Prediction Output

The output file contains:

- Sample ID
- Input text
- Predicted sentiment
- Prediction confidence

The prediction script also logs its execution information and output artifacts to MLflow.

**Note:** Prediction confidence is not a guarantee of correctness. A model can produce a high-confidence prediction that is still incorrect.

---

## 11. Project Structure

```text
MLflow_Champion_Challenger/
│
├── data/
│   ├── eval_inputs.csv
│   └── new_data.csv
│
├── outputs/
│   ├── distilbert_predictions.csv
│   ├── roberta_predictions.csv
│   ├── distilbert_classification_report.json
│   ├── roberta_classification_report.json
│   ├── distilbert_confusion_matrix.png
│   ├── roberta_confusion_matrix.png
│   ├── macro_f1_comparison.png
│   ├── latency_comparison.png
│   ├── disagreement_examples.csv
│   ├── model_comparison.csv
│   ├── model_runs.json
│   ├── metrics_comparison.json
│   ├── metrics_summary.json
│   └── new_data_predictions.csv

│
├── screenshots/
│   ├── 01_mlflow_experiments.png
│   ├── 02_model_registry.png
│   └── 03_new_predictions.png
│
├── evaluate.py
├── register_champion.py
├── register_models.py
├── predict_from_mlflow.py
├── news_sentiment_analysis.csv
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 12. Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.11 | Programming language |
| Hugging Face Transformers | Pretrained sentiment classification models |
| PyTorch | Model inference |
| MLflow | Experiment tracking and model management |
| MLflow Model Registry | Model registration and aliases |
| Pandas | Dataset handling and processing |
| Scikit-learn | Evaluation metrics and classification reports |
| Matplotlib | Evaluation plots |
| Git and GitHub | Version control and project submission |

---

## 13. Installation and Setup

### Step 1: Clone the Repository

```powershell
git clone https://github.com/Mohammed-Zubiyaan-A/1MS25CI081-mlflow-champion-challenger.git
```

### Step 2: Navigate to the Project Directory

```powershell
cd 1MS25CI081-mlflow-champion-challenger
```

### Step 3: Create a Virtual Environment

```powershell
py -3.11 -m venv venv311
```

### Step 4: Activate the Virtual Environment

```powershell
.\venv311\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 14. Running the Project

Run the following commands from the project root directory.

### Step 1: Start the MLflow Tracking Server

Open a separate PowerShell terminal, activate the virtual environment, and execute:

```powershell
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000
```

Open the MLflow UI:

```text
http://127.0.0.1:5000
```

Keep the tracking server running while executing the project scripts.

### Step 2: Run Model Evaluation

In another PowerShell terminal:

```powershell
python evaluate.py
```

This evaluates both models and records the results in MLflow.

### Step 3: Register the Models

```powershell
python register_champion.py
```

This registers both models and assigns the `champion` and `challenger` aliases according to the evaluation decision.

### Step 4: Predict on New Data

```powershell
python predict_from_mlflow.py
```

This loads the Champion model from the MLflow Model Registry and generates predictions on the new dataset.

---

## 15. Screenshots

The `screenshots/` directory contains the following project evidence:

1. **MLflow Experiment Tracking:** Shows the model evaluation runs and comparison run.
2. **MLflow Model Registry:** Shows the registered models and their Champion and Challenger aliases.
3. **New Data Predictions:** Shows the terminal output generated by the MLflow-based prediction script.

---

## 16. Limitations

1. The evaluation dataset contains only 200 balanced English-language samples, so the results may not represent all news articles or sentiment classification scenarios.
2. Both models are pretrained sentiment classifiers and were not specifically fine-tuned on the selected news dataset.
3. Evaluation and inference were performed using CPU hardware, so latency measurements may differ on systems with GPUs or different hardware configurations.
4. The models perform binary sentiment classification and do not support neutral sentiment in this implementation.
5. Prediction confidence does not guarantee prediction correctness. Some sentences may be misclassified despite receiving high confidence scores.

The results should therefore be interpreted as a comparative experiment rather than a guarantee of production-level sentiment classification accuracy.

---
## 17. Declaration of AI Assistance

ChatGPT was used as an AI assistance tool for debugging, understanding MLflow concepts, improving code structure, and preparing project documentation. The implementation was executed and tested by the student, and the reported experimental results were obtained from the actual project runs.

## 18. Conclusion

This project demonstrates a complete Champion–Challenger model comparison workflow using Hugging Face and MLflow.

Two pretrained sentiment classification models were evaluated using identical input data. Their performance was compared using Accuracy, Macro Precision, Macro Recall, Macro-F1, and inference latency.

Although Siebert-RoBERTa achieved a higher Macro-F1 score, its inference latency exceeded the predefined limit. Therefore, DistilBERT was retained as the Champion.

Both models were registered in the MLflow Model Registry, and their aliases were verified. The selected Champion was successfully loaded through MLflow to generate predictions on eight new examples.

The project demonstrates experiment tracking, model evaluation, predefined model selection, model registration, artifact logging, and MLflow-based inference.

---


## 18. GitHub Repository

**Repository:** [1MS25CI081 – MLflow Champion–Challenger](https://github.com/Mohammed-Zubiyaan-A/1MS25CI081-mlflow-champion-challenger)

