import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import shap

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "model.joblib"
FEATURES_PATH = BASE_DIR.parent / "models" / "feature_names.json"
PROFILES_PATH = BASE_DIR.parent / "data" / "processed" / "fighters_profiles.parquet"


def load_profiles():
    if not PROFILES_PATH.exists():
        raise FileNotFoundError(
            f"fighters_profiles.parquet not found at {PROFILES_PATH}"
        )

    df = pd.read_parquet(PROFILES_PATH)

    if "name" in df.columns:
        df["name"] = df["name"].astype(str).str.lower().str.strip()
        df = df.set_index("name")
        return df
    try:
        df.index = df.index.map(lambda x: str(x).lower().strip())
    except:
        pass

    return df


def find_fighter(df, name: str):
    name_norm = name.lower().strip()
    if name_norm in df.index:
        return df.loc[name_norm]

    matches = [idx for idx in df.index if name_norm in idx]
    if len(matches) == 1:
        return df.loc[matches[0]]

    return None


def build_input_vector(f1, f2, feature_order):
    row = {}

    numeric_fields = [
        "Ht.",
        "Wt.",
        "Reach",
        "fight_count",
        "winrate",
        "avg_kd",
        "avg_str",
        "avg_td",
        "avg_sub",
        "avg_ctrl",
        "avg_sig",
        "last_winrate",
        "last_avg_str",
        "elo",
    ]
    finish_fields = [
        "finish_rate",
        "pct_finish_r1",
        "pct_finish_r2",
        "pct_finish_r3p",
        "avg_finish_round",
    ]

    for field in finish_fields:
        r_val = (
            float(f1[field]) if field in f1.index and not pd.isna(f1[field]) else 0.0
        )
        b_val = (
            float(f2[field]) if field in f2.index and not pd.isna(f2[field]) else 0.0
        )

        row[f"R_{field}"] = r_val
        row[f"B_{field}"] = b_val
        row[f"DIFF_{field}"] = r_val - b_val
    for f in numeric_fields:
        v1 = float(f1.get(f, 0) if not pd.isna(f1.get(f, 0)) else 0)
        v2 = float(f2.get(f, 0) if not pd.isna(f2.get(f, 0)) else 0)

        row[f"R_{f}"] = v1
        row[f"B_{f}"] = v2
        row[f"DIFF_{f}"] = v1 - v2

    stances = ["Orthodox", "Southpaw", "Switch", "Sideways", "Unknown"]
    for s in stances:
        row[f"R_Stance_{s}"] = 1 if str(f1.get("Stance", "Unknown")) == s else 0
        row[f"B_Stance_{s}"] = 1 if str(f2.get("Stance", "Unknown")) == s else 0

    weight_classes = [
        "Catch Weight",
        "Featherweight",
        "Flyweight",
        "Heavyweight",
        "Light Heavyweight",
        "Lightweight",
        "Middleweight",
        "Open Weight",
        "Super Heavyweight",
        "Welterweight",
        "Women's Bantamweight",
        "Women's Featherweight",
        "Women's Flyweight",
        "Women's Strawweight",
    ]
    for wc in weight_classes:
        row[f"Weight_Class_{wc}"] = 1 if str(f1.get("Weight_Class", "")) == wc else 0

    styles = ["No Clear Style", "Striker", "Wrestler", "Unknown"]
    for st in styles:
        row[f"R_Fighting Style_{st}"] = (
            1 if str(f1.get("Fighting Style", "Unknown")) == st else 0
        )
        row[f"B_Fighting Style_{st}"] = (
            1 if str(f2.get("Fighting Style", "Unknown")) == st else 0
        )

    X = pd.DataFrame([row])

    for col in feature_order:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_order]
    return X


def main():
    if len(sys.argv) < 3:
        print("Usage:\n  python src/predict.py 'Fighter 1' 'Fighter 2'")
        sys.exit(1)

    f1_name = sys.argv[1]
    f2_name = sys.argv[2]

    print(f"🔍 Predicting: {f1_name} vs {f2_name}")

    profiles = load_profiles()

    f1 = find_fighter(profiles, f1_name)
    f2 = find_fighter(profiles, f2_name)

    if f1 is None:
        print(f"❌ Fighter not found: {f1_name}")
        sys.exit(1)
    if f2 is None:
        print(f"❌ Fighter not found: {f2_name}")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH) as f:
        feature_order = json.load(f)

    X = build_input_vector(f1, f2, feature_order)
    proba = float(model.predict_proba(X)[0][1])
    winner = f1_name if proba >= 0.5 else f2_name

    print("\n🎯 RESULT")
    print(f"🔥 Победит: {winner} ({proba * 100:.1f}%)\n")
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]
        else:
            sv = shap_vals[0]

        shap_series = pd.Series(sv, index=feature_order)
        important = shap_series.abs().sort_values(ascending=False).head(10)

        print("Top contributing features:")
        for feat in important.index:
            print(f" {feat:35s}  SHAP={shap_series[feat]:+.4f}")

    except Exception as e:
        print("⚠️ SHAP explanation failed:", e)


if __name__ == "__main__":
    main()
