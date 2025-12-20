from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
csv_path = BASE / "data/processed/processed.csv"
parquet_path = BASE / "data/processed/processed_feast.parquet"

df = pd.read_csv(csv_path)

# уникальный entity id
df["fight_id"] = df.index.astype("int64")

# encode object → category codes
for col in df.columns:
    if col == "target":
        continue
    if df[col].dtype == "object":
        df[col] = df[col].astype("category").cat.codes

# фиктивный timestamp (для Feast, без историчности)
df["event_timestamp"] = pd.Timestamp("2025-01-01", tz="UTC")

df.to_parquet(parquet_path, index=False)

print("✅ Saved:", parquet_path)
print("Rows:", len(df))
