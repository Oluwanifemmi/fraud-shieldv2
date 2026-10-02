from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String
from feast.value_type import ValueType

customer = Entity(name="cc_num", value_type=ValueType.INT64, join_keys=["cc_num"])

fraud_source = FileSource(
    path="fraud_transaction.parquet",
    timestamp_field="event_timestamp",
)

fraud_feature_view = FeatureView(
    name="fraud_features",
    entities=[customer],
    ttl=timedelta(0),
    schema=[
        Field(name="amt", dtype=Float32),
        Field(name="gender", dtype=String),
        Field(name="prior_txn_count", dtype=Int64),
        Field(name="prior_avg_amt", dtype=Float32),
    ],
    source=fraud_source,
    online=True,
)