from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64
from datetime import timedelta
from feast.data_format import ParquetFormat

personality_entity = Entity(
    name="personality",
    join_keys=["user_id"],
    description="User personality traits",
)

personality_source = FileSource(
    name="personality_data",
    path="/opt/airflow/data/processed/personality_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
    file_format=ParquetFormat(),
)

personality_features = FeatureView(
    name="personality_features",
    entities=[personality_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="user_id", dtype=Int64),
        Field(name="time_broken_spent_alone", dtype=Float32),
        Field(name="stage_fear", dtype=Int64),
        Field(name="social_event_attendance", dtype=Float32),
        Field(name="going_outside", dtype=Float32),
        Field(name="drained_after_socializing", dtype=Int64),
        Field(name="friends_circle_size", dtype=Float32),
        Field(name="post_frequency", dtype=Float32),
        Field(name="personality_encoded", dtype=Int64),
    ],
    source=personality_source,
    online=True,
    tags={"team": "mlops", "dataset": "personality"},
)