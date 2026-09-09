from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import numpy as np
import logging
import time
from datetime import datetime

# Set up logging
logging.basicConfig(
    filename='predictions.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

app = FastAPI(title="Fraud Detection API")

# Load model and feature names once, at startup
model = joblib.load('fraud_model.pkl')
with open('feature_names.json', 'r') as f:
    feature_names = json.load(f)

class Transaction(BaseModel):
    features: dict

@app.get("/")
def read_root():
    return {"message": "Fraud Detection API is running"}

@app.post("/score-transaction")
def score_transaction(transaction: Transaction):
    start_time = time.time()

    input_data = [transaction.features.get(f, 0) for f in feature_names]
    input_array = np.array(input_data).reshape(1, -1)

    fraud_probability = model.predict_proba(input_array)[0][1]
    prediction = int(fraud_probability > 0.5)

    latency_ms = (time.time() - start_time) * 1000

    # Log this prediction
    logging.info(
        f"prediction={'fraud' if prediction == 1 else 'legit'} "
        f"probability={fraud_probability:.6f} "
        f"latency_ms={latency_ms:.2f}"
    )

    return {
        "fraud_probability": float(fraud_probability),
        "prediction": "fraud" if prediction == 1 else "legit",
        "threshold_used": 0.5,
        "latency_ms": round(latency_ms, 2)
    }