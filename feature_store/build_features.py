"""
build_features.py

Builds feature_repo/fraud_transaction.parquet from the VALIDATED,
cleaned data produced by validation.py — not the raw sample.

Run from the project root, after validation.py:
    python feature_store/build_features.py
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
CLEAN_CSV = HERE.parent / "newdata" / "fraudTrain_sample_clean.csv"
OUT_PATH = HERE / "feature_repo" / "fraud_transaction.parquet"


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    if not CLEAN_CSV.exists():
        raise FileNotFoundError(
            f"Cleaned data not found: {CLEAN_CSV}. Run validation.py first."
        )

    df = pd.read_csv(
        CLEAN_CSV,
        usecols=["trans_date_trans_time", "cc_num", "amt", "gender"],
    )
    df["event_timestamp"] = pd.to_datetime(df["trans_date_trans_time"], utc=True)
    df = df.sort_values(["cc_num", "event_timestamp"]).reset_index(drop=True)

    by_card = df.groupby("cc_num")["amt"]
    prior_count = by_card.cumcount()               # transactions before this one, per card
    prior_sum = by_card.cumsum() - df["amt"]        # spend before this one, per card

    df["prior_txn_count"] = prior_count.astype("int64")
    df["prior_avg_amt"] = (
        prior_sum / prior_count.where(prior_count > 0)
    ).astype("float32")
    df["prior_avg_amt"] = df["prior_avg_amt"].fillna(0.0)  # a card's first transaction has no history

    df["amt"] = df["amt"].astype("float32")
    df["cc_num"] = df["cc_num"].astype("int64")
    df = df[["cc_num", "event_timestamp", "amt", "gender", "prior_txn_count", "prior_avg_amt"]]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    logger.info(f"Wrote {len(df):,} rows to {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()