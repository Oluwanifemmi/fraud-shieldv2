"""
train_model.py

Builds a point-in-time-correct training set from the feature store
and trains a simple XGBoost classifier.

Run from the project root, after setup_feast.py:
    python feature_store/train_model.py
"""

import logging
from pathlib import Path

import joblib
import pandas as pd
from feast import FeatureStore
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
REPO_DIR = HERE / "feature_repo"
CLEAN_CSV = HERE.parent / "newdata" / "fraudTrain_sample_clean.csv"
MODEL_OUTPUT_PATH = HERE.parent / "model" / "fraud_model.pkl"

FEATURES = [
    "fraud_features:amt",
    "fraud_features:gender",
    "fraud_features:prior_txn_count",
    "fraud_features:prior_avg_amt",
]
FEATURE_COLS = ["amt", "gender", "prior_txn_count", "prior_avg_amt"]


def build_entity_df() -> pd.DataFrame:
    """cc_num + event_timestamp + label, one row per transaction."""
    df = pd.read_csv(CLEAN_CSV, usecols=["cc_num", "trans_date_trans_time", "is_fraud"])
    df["event_timestamp"] = pd.to_datetime(df["trans_date_trans_time"], utc=True)
    df["cc_num"] = df["cc_num"].astype("int64")
    return df[["cc_num", "event_timestamp", "is_fraud"]]


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    entity_df = build_entity_df()
    logger.info(f"Entity dataframe: {len(entity_df):,} rows, {entity_df['is_fraud'].sum()} fraud")

    store = FeatureStore(repo_path=str(REPO_DIR))
    training_df = store.get_historical_features(entity_df=entity_df, features=FEATURES).to_df()
    logger.info(f"Training dataframe: {training_df.shape}")

    gender_encoder = LabelEncoder()
    training_df["gender"] = gender_encoder.fit_transform(training_df["gender"])

    X = training_df[FEATURE_COLS]
    y = training_df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    logger.info(f"Test AUC: {auc:.4f}")

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "gender_encoder": gender_encoder, "feature_cols": FEATURE_COLS},
        MODEL_OUTPUT_PATH,
    )
    logger.info(f"Saved model bundle to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()