from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.types import Int64, String

# Оффлайн источник с подготовленными признаками
detox_source = FileSource(
    path="../data/features/detox_features.csv",
    timestamp_field="event_timestamp",
)

# Entity для связывания строк набора данных
sample = Entity(name="sample_id", join_keys=["sample_id"])

detox_view = FeatureView(
    name="detox_features",
    entities=[sample],
    ttl=timedelta(days=365),
    schema=[
        Field(name="split", dtype=String),
        Field(name="input_text", dtype=String),
        Field(name="target_text", dtype=String),
    ],
    source=detox_source,
)

detox_service = FeatureService(name="detox_service", features=[detox_view])
