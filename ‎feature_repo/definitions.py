from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int32, Int64, String
from feast.value_type import ValueType
# 1. Сущность (ключ для объединения)
currency_entity = Entity(
    name="currency_record",
    join_keys=["record_id"],
    value_type=ValueType.INT64,
    description="Уникальный ID записи о курсе валют",
)

# 2. Источник данных - ВАШИ данные
# Измените путь на:
currency_source = FileSource(
    name="currency_data",
    path="data/currency_data.parquet",  # ← файл ВНУТРИ feature_repo!
    timestamp_field="date",
    created_timestamp_column="created_at",
)

# 3. Feature View с ключевыми признаками
currency_features = FeatureView(
    name="currency_features",
    entities=[currency_entity],
    ttl=timedelta(days=365),
    schema=[
            Field(name="USD_RUB", dtype=Float32),
    Field(name="EUR_RUB", dtype=Float32),
    Field(name="GBP_RUB", dtype=Float32),
    Field(name="date", dtype=Int64),
    Field(name="day_of_week", dtype=Int64),
    Field(name="is_weekend", dtype=Int64),
    Field(name="departure_hour_bucket", dtype=String),
    Field(name="currency_pair", dtype=String),
    Field(name="created_at", dtype=Int64),
    Field(name="record_id", dtype=Int64),
    ],
    source=currency_source,
    online=True,
)