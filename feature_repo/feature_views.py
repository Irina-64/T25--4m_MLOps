from datetime import timedelta
from feast import FeatureView, Field
from feast.types import Float32, Int64
from feast.infra.offline_stores.file_source import FileSource
from feast.value_type import ValueType
from .entities import user

# Определяем источник данных (offline таблица)
# Путь относительно корня проекта или абсолютный путь
user_features_source = FileSource(
    name="user_features_source",
    path="data/processed/feast_features.csv",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Feature View с признаками пользователя
user_features_fv = FeatureView(
    name="user_features",
    entities=[user],
    ttl=timedelta(days=365),
    schema=[
        Field(name="all_sum", dtype=Float32),
        Field(name="all_mean", dtype=Float32),
        Field(name="all_std", dtype=Float32),
        Field(name="all_count", dtype=Float32),
        Field(name="all_max", dtype=Float32),
        Field(name="all_min", dtype=Float32),
        Field(name="last3m_sum", dtype=Float32),
        Field(name="last3m_mean", dtype=Float32),
        Field(name="last3m_std", dtype=Float32),
        Field(name="last3m_count", dtype=Float32),
        Field(name="last3m_max", dtype=Float32),
        Field(name="last3m_min", dtype=Float32),
        Field(name="last1m_sum", dtype=Float32),
        Field(name="last1m_mean", dtype=Float32),
        Field(name="last1m_std", dtype=Float32),
        Field(name="last1m_count", dtype=Float32),
        Field(name="last1m_max", dtype=Float32),
        Field(name="last1m_min", dtype=Float32),
        Field(name="active_days", dtype=Float32),
        Field(name="avg_gap_days", dtype=Float32),
        Field(name="max_gap_days", dtype=Float32),
        Field(name="days_since_last", dtype=Float32),
        Field(name="income_share", dtype=Float32),
        Field(name="income_days", dtype=Float32),
        Field(name="outcome_days", dtype=Float32),
        Field(name="has_gap_30d", dtype=Int64),
        Field(name="weekend_ratio", dtype=Float32),
        Field(name="activity_trend_sep_nov", dtype=Float32),
    ],
    source=user_features_source,
    tags={"team": "mlops"},
)

