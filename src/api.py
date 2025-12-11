from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import os

app = FastAPI()


def _load_model():
    # Prefer explicit models/model.joblib if present, else pick latest .joblib in models/
    model_dir = "models"
    if os.path.exists(os.path.join(model_dir, "model.joblib")):
        return joblib.load(os.path.join(model_dir, "model.joblib"))

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory '{model_dir}' not found")

    candidates = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.joblib')]
    if not candidates:
        raise FileNotFoundError(f"No .joblib models found in '{model_dir}'")

    # pick most recently modified
    latest = max(candidates, key=os.path.getmtime)
    return joblib.load(latest)


try:
    model = _load_model()
    _MODEL_PATH = getattr(model, '_loaded_from', None)
except Exception as e:
    model = None
    _load_error = str(e)


@app.post("/predict")
def predict(payload: dict):
    """Simple prediction endpoint.

    Expects a JSON object that maps feature names to values. Example:
    {"feature1": 1.0, "feature2": "A", ...}

    Returns:
        {"delay_prob": 0.123}
    """
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {_load_error}")

    # Build DataFrame from payload
    try:
        df = pd.DataFrame([payload])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")

    # Try predict_proba; if feature alignment required, attempt to align using feature_names_in_
    try:
        preds = model.predict_proba(df)[:, 1]
    except Exception as e:
        # Attempt to align features if model exposes feature names
        if hasattr(model, 'feature_names_in_'):
            cols = list(model.feature_names_in_)
            for c in cols:
                if c not in df.columns:
                    df[c] = 0
            df = df[cols]
            try:
                preds = model.predict_proba(df)[:, 1]
            except Exception as e2:
                raise HTTPException(status_code=400, detail=f"Prediction failed after aligning features: {e2}")
        else:
            raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    return {"delay_prob": float(preds[0])}
