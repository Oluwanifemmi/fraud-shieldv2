"""
train.py

Kubernetes Job: builds features from the FULL raw dataset, registers and
materializes them with Feast, trains XGBoost, and writes the model bundle
to the mounted output volume.
"""

import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from feast import FeatureStore
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

RAW_CSV = Path("data/fraudTrain.csv")
REPO_DIR = Path("feature_store/feature_repo")
PARQUET_PATH = REPO_DIR / "fraud_transaction.parquet"
MODEL_OUTPUT_PATH = Path("/output/fraud_model.pkl")

sys.path.insert(0, str(REPO_DIR))
from fraud_features import customer, fraud_feature_view  # noqa: E402

FEATURES = [
    "fraud_features:amt",
    "fraud_features:gender",
    "fraud_features:prior_txn_count",
    "fraud_features:prior_avg_amt",
]
FEATURE_COLS = ["amt", "gender", "prior_txn_count", "prior_avg_amt"]


def build_feature_parquet() -> pd.DataFrame:
    """Same logic as build_features.py, run here against the full raw file."""
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
    df["cc_num"] = df["cc_num"].astype("int64")

    feature_cols_df = df[["cc_num", "event_timestamp", "amt", "gender", "prior_txn_count", "prior_avg_amt"]]
    feature_cols_df.to_parquet(PARQUET_PATH, index=False)
    logger.info(f"Wrote {len(feature_cols_df):,} rows to {PARQUET_PATH}")

    return df[["cc_num", "event_timestamp", "is_fraud"]]


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    entity_df = build_feature_parquet()
    logger.info(f"Entity dataframe: {len(entity_df):,} rows, {entity_df['is_fraud'].sum()} fraud")

    store = FeatureStore(repo_path=str(REPO_DIR))
    store.apply([customer, fraud_feature_view])
    logger.info("Registered entity and feature view")

    start = entity_df["event_timestamp"].min().to_pydatetime()
    end = (entity_df["event_timestamp"].max() + pd.Timedelta(seconds=1)).to_pydatetime()
    store.materialize(start_date=start, end_date=end)
    logger.info("Materialized features into the online store")

    training_df = store.get_historical_features(entity_df=entity_df, features=FEATURES).to_df()
    logger.info(f"Training dataframe: {training_df.shape}")

    gender_encoder = LabelEncoder()
    training_df["gender"] = gender_encoder.fit_transform(training_df["gender"])

    X = training_df[FEATURE_COLS]
    y = training_df["is_fraud"]

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