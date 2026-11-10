# Champion–Challenger Model Comparison with Hugging Face and MLflow

## 1. Project Overview

This project compares two pretrained Hugging Face sentiment classification models using MLflow. Both models are evaluated on the same fixed evaluation dataset to determine which model performs better.

## 2. Models Selected

| Model | Hugging Face Model ID | Role |
|---|---|---|
| Model A | distilbert/distilbert-base-uncased-finetuned-sst-2-english | Initial Champion |
| Model B | siebert/sentiment-roberta-large-english | Challenger |

## 3. Dataset

**Original dataset:** news_sentiment_analysis.csv

**Evaluation dataset:** data/eval_inputs.csv

- Total evaluation samples: 200
- Positive samples: 100
- Negative samples: 100
- Language: English
- Random seed: 42

Both models will receive exactly the same evaluation inputs.

Neutral samples are excluded because both selected models are binary sentiment classifiers.

## 4. Champion Selection Rule

The primary evaluation metric is Macro-F1.

The initial Champion is Model A (DistilBERT), and Model B (Siebert RoBERTa) is the Challenger.

The Challenger will replace the Champion when:

1. Its Macro-F1 score improves by at least 0.01 absolute.
2. Its average inference latency does not exceed five times the Champion's average latency.

If the absolute Macro-F1 difference is less than 0.01, the faster model will be selected.

If the Challenger performs more than 0.01 worse, the existing Champion will be retained.

If the Challenger exceeds the latency limit while showing a qualifying Macro-F1 improvement, the existing Champion will be retained.

This decision rule is defined before the final experiment runs.

## 5. Evaluation Metrics

- Accuracy
- Precision
- Recall
- Macro-F1
- Average inference latency
- Total inference time

## 6. Experiment Tracking

MLflow will be used to track:

- Model parameters
- Evaluation metrics
- Model artifacts
- Prediction outputs
- Confusion matrices
- Classification reports
- Model comparison results

Both models will be registered under the same MLflow registered model name, with aliases assigned according to the final comparison.

## 7. Technology Stack

- Python
- Hugging Face Transformers
- PyTorch
- MLflow
- Pandas
- Scikit-learn
- Matplotlib
## Experimental Results

Both models were evaluated using the same 200 English-language
news samples, consisting of 100 positive and 100 negative samples.

| Metric | DistilBERT | Siebert-RoBERTa |
|---|---:|---:|
| Accuracy | 0.5950 | 0.6300 |
| Macro Precision | 0.5951 | 0.6578 |
| Macro Recall | 0.5950 | 0.6300 |
| Macro F1 | 0.5949 | 0.6129 |
| Average Latency (ms/sample) | 129.6276 | 900.7777 |
| Total Inference Time (seconds) | 25.9255 | 180.1555 |

## Champion Selection

DistilBERT was retained as the Champion.

Although Siebert-RoBERTa achieved a Macro-F1 improvement of
0.018, its average inference latency was approximately 6.95
times higher than DistilBERT.

Since the predefined selection rule limits challenger latency
to five times the champion's latency, Siebert-RoBERTa did not
qualify for promotion.

Therefore, DistilBERT remains the Champion.