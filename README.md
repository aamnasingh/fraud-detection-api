# Real-Time Fraud Detection System

A machine learning system that detects fraudulent credit card transactions, wrapped in a production-style REST API and containerized with Docker. Built to explore how fraud detection actually works at a systems level — not just as a notebook exercise, but as something that could plausibly run in production.

## Problem

Credit card fraud detection is a classic extreme class-imbalance problem: in the dataset used here, only **0.173%** of transactions (492 out of 284,807) are fraudulent. A naive model can achieve 99.8% accuracy by never predicting fraud at all — which makes accuracy a useless metric and the real modeling challenge about precision/recall tradeoffs, not raw correctness.

## Data

[Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) — 284,807 anonymized European card transactions over 48 hours. Features `V1`–`V28` are PCA-transformed for confidentiality; `Amount` and `Time` are the only interpretable raw features.

**Key EDA findings:**
- Fraud and legitimate transaction amounts are both heavily right-skewed, but fraud amounts cap out much lower (~2,000) than legitimate ones (~25,000) — suggesting fraudsters keep amounts smaller to avoid detection thresholds.
- Legitimate transactions follow a clear day/night volume cycle; fraud transactions are more evenly scattered across time, without the same rhythm.
- Simple correlation analysis flagged `V17`, `V14`, `V12`, `V10` as the strongest fraud indicators — a finding that later held up under SHAP analysis (see below).

## Approach

### 1. Baseline models
Logistic Regression and Random Forest were trained first, with **PR-AUC (not accuracy)** as the primary metric, since PR-AUC is far more informative than ROC-AUC under severe imbalance.

### 2. Handling class imbalance
Three approaches were compared head-to-head on the same held-out test set:

| Approach | Recall (fraud) | Precision (fraud) | PR-AUC | False Alarms |
|---|---|---|---|---|
| Random Forest (baseline) | 81.63% | 94.12% | 0.8647 | 5 |
| Random Forest + SMOTE | 81.63% | 82.47% | 0.8698 | 17 |
| Random Forest + Class Weighting | 80.61% | 91.86% | 0.8567 | 7 |
| **XGBoost + scale_pos_weight** | **84.69%** | **87.37%** | **0.8771** | 12 |

**Notably, SMOTE oversampling did not meaningfully outperform the baseline** — it gave a marginal PR-AUC gain but more than tripled false alarms (5 → 17) for identical recall. This is reported honestly rather than omitted, since a well-behaved baseline sometimes really is competitive with fancier resampling techniques. XGBoost with `scale_pos_weight` was the clear winner and was carried forward.

### 3. Hyperparameter tuning
`RandomizedSearchCV` (15 candidates × 3-fold CV, optimizing PR-AUC) found best parameters (`n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7`). Tuning gave only a marginal improvement (PR-AUC 0.8771 → 0.8784), trading a bit of recall for a bit of precision — suggesting the default configuration was already close to optimal, and further gains would more likely come from better features than further tuning.

**Final model: Tuned XGBoost — Precision 88.0%, Recall 82.7%, PR-AUC 0.878.**

### 4. Explainability (SHAP)
SHAP analysis validated the earlier correlation findings: `V14`, `V10`, and `V12` ranked among the top global features, with `V4` emerging as the single most important feature — despite only moderate linear correlation, suggesting XGBoost captured a non-linear relationship a simple correlation check missed. Individual-prediction force plots confirmed the model's decisions are traceable to consistent, specific feature patterns rather than arbitrary combinations — an important property for any system that may need to justify a flagged transaction to a compliance team or customer.

## Deployment

The final model is served via a **FastAPI** REST endpoint (`POST /score-transaction`), returning a fraud probability, binary prediction, and prediction latency. The service is containerized with **Docker** for portability, and every prediction is logged (timestamp, prediction, probability, latency) for basic production monitoring.

**Verification:** the containerized API was tested against both a known-legitimate feature pattern (0.007% fraud probability) and a known-fraud pattern derived from the SHAP analysis (99.95% fraud probability), confirming consistent behavior outside the original notebook environment.

One real observation from testing: the **first** request after container startup took 69ms, while subsequent requests took only 3–7ms — likely one-time initialization overhead inside XGBoost/scikit-learn. In a real deployment this would be handled with a warm-up request immediately after container start.

### Tech stack
`Python · pandas · scikit-learn · XGBoost · SHAP · FastAPI · Docker`

## Project structure
```
fraud-detection-project/
├── fraud_detection.ipynb      # EDA, modeling, tuning, SHAP analysis
├── main.py                    # FastAPI application
├── fraud_model.pkl            # Trained, tuned XGBoost model
├── feature_names.json         # Expected feature order for inference
├── requirements.txt           # API dependencies (trimmed to production needs)
├── Dockerfile
└── predictions.log            # Runtime prediction log (generated at runtime)
```

## Running it locally

```bash
# Build and run with Docker
docker build -t fraud-detection-api .
docker run -p 8000:8000 fraud-detection-api

# Interactive API docs
# → http://127.0.0.1:8000/docs
```

## What I'd do with more time
- **Persist logs outside the container** via a mounted Docker volume — currently logs live inside the container's filesystem and are lost if the container is removed.
- **Feature engineering over further tuning** — given how little hyperparameter tuning moved the needle, the next real lever is likely engineered features (e.g., transaction velocity, merchant-level aggregates) rather than more parameter search.
- **Streaming/real-time scoring** — simulate a Kafka or Redis Streams pipeline feeding transactions to the API continuously, closer to how a live fraud system would actually operate.
- **Model monitoring for drift** — track the distribution of predicted probabilities over time to catch when the model's behavior starts to shift from what it saw in training.

## Author
**Aamna Singh** — [GitHub](https://github.com/aamnasingh)