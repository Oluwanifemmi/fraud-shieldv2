"""
validation.py

Validates the raw fraud transactions sample against checks on every column
before it's used to build features. Writes:
  - expectation/expectation_suite.json  (the checks, for reference)
  - expectation/validation_results.txt  (pass/fail summary)
  - expectation/failed.csv              (rows that failed any check)
  - newdata/fraudTrain_sample_clean.csv (rows that passed every check)
"""

import json
import os

import pandas as pd
from great_expectations.dataset import PandasDataset

RAW_PATH = "newdata/fraudTrain_sample.csv"
EXPECTATION_DIR = "data_validation/expectation"
CLEAN_PATH = "newdata/fraudTrain_sample_clean.csv"

VALID_GENDERS = ["M", "F"]
VALID_CATEGORIES = ["grocery_pos", "gas_transport", "shopping_net", "misc_pos", "entertainment", "food_dining"]

os.makedirs(EXPECTATION_DIR, exist_ok=True)
os.makedirs("newdata", exist_ok=True)

# Load data
df = pd.read_csv(RAW_PATH)

print("\n---------------------------------------")
print("Data Validation")
print(df.head())
print(df.columns.tolist())

# Great Expectations dataset
gx_df = PandasDataset(df)

# Record of the checks being run (for reference only)
expectation_suite = {
    "trans_num": {"unique": True, "not_null": True},
    "trans_date_trans_time": {"not_null": True},
    "cc_num": {"not_null": True, "min": 1},
    "merchant": {"not_null": True},
    "category": {"in_set": VALID_CATEGORIES},
    "amt": {"min": 0, "max": 10000},
    "gender": {"in_set": VALID_GENDERS},
    "lat": {"min": -90, "max": 90},
    "long": {"min": -180, "max": 180},
    "city_pop": {"min": 1},
    "merch_lat": {"min": -90, "max": 90},
    "merch_long": {"min": -180, "max": 180},
    "is_fraud": {"min": 0, "max": 1, "not_null": True},
}
with open(f"{EXPECTATION_DIR}/expectation_suite.json", "w") as f:
    json.dump(expectation_suite, f, indent=4)
print("Expectation suite saved")

checks = {}

print("\n--------------------------------- trans_num unique")
checks["trans_num_unique"] = gx_df.expect_column_values_to_be_unique("trans_num").success

print("\n--------------------------------- trans_num not null")
checks["trans_num_not_null"] = gx_df.expect_column_values_to_not_be_null("trans_num").success

print("\n--------------------------------- trans_date_trans_time not null")
checks["trans_date_not_null"] = gx_df.expect_column_values_to_not_be_null("trans_date_trans_time").success

print("\n--------------------------------- cc_num not null")
checks["cc_num_not_null"] = gx_df.expect_column_values_to_not_be_null("cc_num").success

print("\n--------------------------------- cc_num positive")
checks["cc_num_positive"] = gx_df.expect_column_values_to_be_between("cc_num", min_value=1, max_value=None).success

print("\n--------------------------------- merchant not null")
checks["merchant_not_null"] = gx_df.expect_column_values_to_not_be_null("merchant").success

print("\n--------------------------------- category in set")
checks["category_in_set"] = gx_df.expect_column_values_to_be_in_set("category", VALID_CATEGORIES).success

print("\n--------------------------------- amt range")
checks["amt_range"] = gx_df.expect_column_values_to_be_between("amt", min_value=0, max_value=10000).success

print("\n--------------------------------- gender values")
checks["gender_in_set"] = gx_df.expect_column_values_to_be_in_set("gender", VALID_GENDERS).success

print("\n--------------------------------- lat range")
checks["lat_range"] = gx_df.expect_column_values_to_be_between("lat", min_value=-90, max_value=90).success

print("\n--------------------------------- long range")
checks["long_range"] = gx_df.expect_column_values_to_be_between("long", min_value=-180, max_value=180).success

print("\n--------------------------------- city_pop positive")
checks["city_pop_positive"] = gx_df.expect_column_values_to_be_between("city_pop", min_value=1, max_value=None).success

print("\n--------------------------------- merch_lat range")
checks["merch_lat_range"] = gx_df.expect_column_values_to_be_between("merch_lat", min_value=-90, max_value=90).success

print("\n--------------------------------- merch_long range")
checks["merch_long_range"] = gx_df.expect_column_values_to_be_between("merch_long", min_value=-180, max_value=180).success

print("\n--------------------------------- is_fraud range")
checks["is_fraud_range"] = gx_df.expect_column_values_to_be_between("is_fraud", min_value=0, max_value=1).success

print("\n--------------------------------- is_fraud not null")
checks["is_fraud_not_null"] = gx_df.expect_column_values_to_not_be_null("is_fraud").success

# Validation summary
print("\nValidation check")
for check, status in checks.items():
    print(f"[{'passed' if status else 'failed'}] {check}")

with open(f"{EXPECTATION_DIR}/validation_results.txt", "w") as f:
    for check, status in checks.items():
        f.write(f"[{'passed' if status else 'failed'}] {check}\n")
print("\nValidation results saved")

# Row-level checks, matching every expectation above
invalid_mask = (
    df["trans_num"].isnull()
    | df["trans_num"].duplicated(keep=False)
    | df["trans_date_trans_time"].isnull()
    | df["cc_num"].isnull()
    | (df["cc_num"] < 1)
    | df["merchant"].isnull()
    | (~df["category"].isin(VALID_CATEGORIES))
    | (df["amt"] < 0)
    | (df["amt"] > 10000)
    | (~df["gender"].isin(VALID_GENDERS))
    | (df["lat"] < -90) | (df["lat"] > 90)
    | (df["long"] < -180) | (df["long"] > 180)
    | (df["city_pop"] < 1)
    | (df["merch_lat"] < -90) | (df["merch_lat"] > 90)
    | (df["merch_long"] < -180) | (df["merch_long"] > 180)
    | (df["is_fraud"] < 0) | (df["is_fraud"] > 1)
    | df["is_fraud"].isnull()
)
invalid_rows = df[invalid_mask]
clean_df = df[~invalid_mask]

print(f"\n{len(invalid_rows)} invalid rows, {len(clean_df)} clean rows")

invalid_rows.to_csv(f"{EXPECTATION_DIR}/failed.csv", index=False)
print("Failed records saved")

clean_df.to_csv(CLEAN_PATH, index=False)
print(f"Clean data saved to {CLEAN_PATH}")