import os

import numpy as np
import pandas as pd

os.makedirs("data/processed", exist_ok=True)

fighters = pd.read_csv("data/raw/Fighters.csv")
fights = pd.read_csv("data/raw/Fights.csv")
events = pd.read_csv("data/raw/Events.csv")
fstats = pd.read_csv("data/raw/Fstats.csv")


def normalize_result(s):
    s = s.astype(str).str.lower().str.strip()
    s = s.replace(
        {
            "w": "win",
            "l": "loss",
            "won": "win",
            "lost": "loss",
            "no contest": "nc",
            "no-contest": "nc",
            "no_contest": "nc",
        }
    )
    return s


r1 = normalize_result(fights["Result_1"])
r2 = normalize_result(fights["Result_2"])
is_draw_nc = r1.isin(["draw", "nc"]) | r2.isin(["draw", "nc"])
target = np.where(r1.eq("win"), 1, np.where(r2.eq("win"), 0, np.nan))
fights["target"] = target
fights.loc[is_draw_nc, "target"] = np.nan
fights = fights.dropna(subset=["target"]).copy()
fights["target"] = fights["target"].astype(int)
print("✅ After target derivation:", fights.shape)

fighters["Full Name"] = fighters["Full Name"].astype(str).str.lower().str.strip()
fights["_R_key"] = fights["Fighter_1"].astype(str).str.lower().str.strip()
fights["_B_key"] = fights["Fighter_2"].astype(str).str.lower().str.strip()


def pick_cols(prefix, df, key_col):
    num = df.select_dtypes(include=["number"]).columns.tolist()
    cat = [c for c in df.columns if df[c].dtype == "object" and c != key_col][:6]
    keep = [c for c in (num[:20] + cat) if c != key_col]
    sub = df[[key_col] + keep].copy()
    sub = sub.rename(columns={c: f"{prefix}_{c}" for c in keep})
    return sub


R = pick_cols("R", fighters, "Full Name")
B = pick_cols("B", fighters, "Full Name")
R = R.rename(columns={"Full Name": "_R_key"})
B = B.rename(columns={"Full Name": "_B_key"})

df = fights.merge(R, on="_R_key", how="left").merge(B, on="_B_key", how="left")
df = df.drop(columns=["_R_key", "_B_key"], errors="ignore")
print("✅ Modeling table shape after fighter merge:", df.shape)

na_frac = df.isna().mean()
drop_cols = na_frac[na_frac > 0.5].index.tolist()
df = df.drop(columns=drop_cols, errors="ignore")

num_cols = [c for c in df.select_dtypes(include=["number"]).columns if c != "target"]
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())

cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
for c in cat_cols:
    df[c] = df[c].fillna("Unknown")

df = pd.get_dummies(df, drop_first=True)
out_path = "data/processed/processed.csv"
df.to_csv(out_path, index=False)
print(f"✅ Финальный датасет сохранён: {df.shape}")
