from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64
from feast.value_type import ValueType
from datetime import timedelta

# -------- ENTITY --------
customer = Entity(
    name="customer_id",
    join_keys=["customer_id"],
    value_type=ValueType.INT64,
)

# -------- SOURCE --------
telco_source = FileSource(
    path="data/telco_features.parquet",
    timestamp_field="event_timestamp",
)

# -------- FEATURE VIEW --------
telco_features = FeatureView(
    name="telco_features",
    entities=[customer],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="SeniorCitizen", dtype=Int64),
        Field(name="tenure", dtype=Int64),
        Field(name="MonthlyCharges", dtype=Float32),
        Field(name="TotalCharges", dtype=Float32),
    ],
    online=False,
    source=telco_source,
)