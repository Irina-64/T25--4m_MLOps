import pandas as pd
from entities import fight  # <-- ВАЖНО: без точки
from feast import FeatureView, Field
from feast.data_format import ParquetFormat
from feast.infra.offline_stores.file_source import FileSource
from feast.types import Float32, Int64

# -------------------------
# Load parquet schema
# -------------------------
PARQUET_PATH = "../data/processed/processed_feast.parquet"
df = pd.read_parquet(PARQUET_PATH)

feature_fields = []

for col in df.columns:
    if col in ("event_timestamp", "target", "fight_id"):
        continue

    dtype = df[col].dtype

    if str(dtype).startswith(("int", "uint", "bool")):
        feast_type = Int64
    else:
        feast_type = Float32

    feature_fields.append(Field(name=col, dtype=feast_type))

print(f"🧩 Loaded {len(feature_fields)} features")

# -------------------------
# Source
# -------------------------
ufc_source = FileSource(
    path=PARQUET_PATH,
    timestamp_field="event_timestamp",
    file_format=ParquetFormat(),
)

# -------------------------
# Feature View
# -------------------------
ufc_features = FeatureView(
    name="ufc_features",
    entities=[fight],
    ttl=None,
    schema=feature_fields,
    online=True,
    source=ufc_source,
)
