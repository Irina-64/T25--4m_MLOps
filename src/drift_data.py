import numpy as np
import pandas as pd

INPUT_PATH = "data/processed/train.csv"
OUTPUT_PATH = "data/processed/prod_recent.csv"

df = pd.read_csv(INPUT_PATH)

# 1️⃣ Берём только реальные числовые НЕ one-hot признаки
feature_cols = [
    c
    for c in df.columns
    if c != "target"
    and (pd.api.types.is_float_dtype(df[c]) or pd.api.types.is_integer_dtype(df[c]))
    and not c.startswith("Weight_Class_")
    and not c.startswith("R_Fighting Style")
    and not c.startswith("B_Fighting Style")
    and not c.endswith("_id")
    and "id" not in c.lower()
]

print("Using features for drift simulation:")
for c in feature_cols:
    print("  ", c)

# 2️⃣ ИСКУССТВЕННЫЙ DRIFT (смещение распределений)
for col in feature_cols:
    df[col] = df[col] * np.random.uniform(1.2, 1.6)

# 3️⃣ Берём подвыборку как "recent production"
df.sample(min(300, len(df))).to_csv(OUTPUT_PATH, index=False)

print(f"\nDrifted production data saved to {OUTPUT_PATH}")
