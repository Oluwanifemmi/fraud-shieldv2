from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from feast import FeatureStore
from pydantic import BaseModel

REPO = Path(__file__).resolve().parents[1] / "feature_store" / "feature_repo"
store = FeatureStore(repo_path=str(REPO))  # created once at startup, like the model

bundle = joblib.load("model/fraud_model.pkl")
model = bundle["model"]
gender_encoder = bundle["gender_encoder"]
feature_cols = bundle["feature_cols"]

app = FastAPI()


class Txn(BaseModel):
    cc_num: int
    amt: float
    gender: str  # "M" or "F", same as training


@app.post("/predict")
def predict(txn: Txn):
    feats = store.get_online_features(
        features=["fraud_features:prior_txn_count", "fraud_features:prior_avg_amt"],
        entity_rows=[{"cc_num": txn.cc_num}],
    ).to_dict()

    prior_txn_count = feats["prior_txn_count"][0]
    prior_avg_amt = feats["prior_avg_amt"][0]

    if prior_txn_count is None:
        # Unknown card: no history in the online store. Treat as a brand-new card.
        prior_txn_count = 0
        prior_avg_amt = 0.0

    gender_encoded = gender_encoder.transform([txn.gender])[0]

    row = pd.DataFrame([{
        "amt": txn.amt,
        "gender": gender_encoded,
        "prior_txn_count": prior_txn_count,
        "prior_avg_amt": prior_avg_amt,
    }])[feature_cols]  # enforce the exact column order used at training time

    probability = float(model.predict_proba(row)[0][1])

    return {
        "fraud_probability": round(probability, 6)
    }