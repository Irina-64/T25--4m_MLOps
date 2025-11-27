from feast import FeatureView, Field, FileSource
from feast.types import Float32, Int64
from datetime import timedelta
from .entity import user_id

user_features_source = FileSource(
    path="data/features.parquet",
    timestamp_field="event_timestamp",
)

user_features = FeatureView(
    name="user_features",
    entities=[user_id],
    ttl=timedelta(days=365),
    schema=[
        Field(name="amount_sum", dtype=Float32),
        Field(name="amount_count", dtype=Int64),
    ],
    source=user_features_source,
    online=True,
)
