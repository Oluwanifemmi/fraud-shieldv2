"""
setup_feast.py

Registers the feature definitions in feature_repo/fraud_features.py,
materializes them into the online store, and runs a sanity lookup.

Run from the project root:
    python feature_store/setup_feast.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd
from feast import FeatureStore

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
REPO_DIR = HERE / "feature_repo"
PARQUET_PATH = REPO_DIR / "fraud_transaction.parquet"

sys.path.insert(0, str(REPO_DIR))
from fraud_features import customer, fraud_feature_view 


def load_source() -> pd.DataFrame:
    """Read the columns the feature view needs; fails loudly if any are missing."""
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"Feature source not found: {PARQUET_PATH}. Run build_features.py first."
        )
    return pd.read_parquet(
        PARQUET_PATH,
        columns=["cc_num", "event_timestamp", "amt", "gender", "prior_txn_count", "prior_avg_amt"],
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    df = load_source()
    logger.info(f"Source has {len(df):,} rows, {df['cc_num'].nunique():,} cards")

    store = FeatureStore(repo_path=str(REPO_DIR))
    store.apply([customer, fraud_feature_view])
    logger.info("Registered entity and feature view")

    start = df["event_timestamp"].min().to_pydatetime()
    end = (df["event_timestamp"].max() + pd.Timedelta(seconds=1)).to_pydatetime()
    store.materialize(start_date=start, end_date=end)
    logger.info("Materialized features into the online store")

    sample_card = int(df["cc_num"].iloc[0])
    online = store.get_online_features(
        features=[
            "fraud_features:amt",
            "fraud_features:gender",
            "fraud_features:prior_txn_count",
            "fraud_features:prior_avg_amt",
        ],
        entity_rows=[{"cc_num": sample_card}],
    ).to_dict()
    logger.info(f"Online lookup for card {sample_card}: {online}")


if __name__ == "__main__":
    main()