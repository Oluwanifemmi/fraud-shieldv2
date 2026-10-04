"""
train.py

Kubernetes Job: builds features directly from the full raw dataset with
pandas, trains XGBoost, and writes the model bundle to the mounted output volume.
"""

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

RAW_CSV = Path("data/fraudTrain.csv")
MODEL_OUTPUT_PATH = Path("/output/fraud_model.pkl")
FEATURE_COLS = ["amt", "gender", "prior_txn_count", "prior_avg_amt"]


def build_features() -> pd.DataFrame:
    df = pd.read_csv(
        RAW_CSV,
        usecols=["trans_date_trans_time", "cc_num", "amt", "gender", "is_fraud"],
    )
    df["event_timestamp"] = pd.to_datetime(df["trans_date_trans_time"], utc=True)
    df = df.sort_values(["cc_num", "event_timestamp"]).reset_index(drop=True)

    by_card = df.groupby("cc_num")["amt"]
    prior_count = by_card.cumcount()
    prior_sum = by_card.cumsum() - df["amt"]

    df["prior_txn_count"] = prior_count.astype("int64")
    df["prior_avg_amt"] = (prior_sum / prior_count.where(prior_count > 0)).astype("float32")
    df["prior_avg_amt"] = df["prior_avg_amt"].fillna(0.0)
    df["amt"] = df["amt"].astype("float32")

    return df[["amt", "gender", "prior_txn_count", "prior_avg_amt", "is_fraud"]]


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    df = build_features()
    logger.info(f"Built features: {len(df):,} rows, {df['is_fraud'].sum()} fraud")

    gender_encoder = LabelEncoder()
    df["gender"] = gender_encoder.fit_transform(df["gender"])

    X = df[FEATURE_COLS]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, eval_metric="logloss")
    model.fit(X_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    logger.info(f"Test AUC: {auc:.4f}")

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "gender_encoder": gender_encoder, "feature_cols": FEATURE_COLS},
        MODEL_OUTPUT_PATH,
    )
    logger.info(f"Saved model bundle to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()