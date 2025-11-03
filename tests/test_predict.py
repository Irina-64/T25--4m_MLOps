import requests
import joblib
import os
import pandas as pd

#URL FastAPI
API_URL = "http://localhost:8080/predict"

#пути к модели и CSV с фичами
MODEL_PATHS = [
    "/workspace/models/lgb_model.joblib",
    "./models/lgb_model.joblib"
]
FEATURES_PATHS = [
    "/workspace/data/processed/processed.csv",
    "./data/processed/processed.csv"
]

#Находим модель
MODEL_PATH = next((p for p in MODEL_PATHS if os.path.exists(p)), None)
if MODEL_PATH is None:
    raise FileNotFoundError("Модель не найдена ни по одному из путей: " + ", ".join(MODEL_PATHS))
model = joblib.load(MODEL_PATH)

#получаем признаки модели
if hasattr(model, "feature_name_") and model.feature_name_ is not None:
    MODEL_FEATURES = list(model.feature_name_)
else:
    #берём все колонки из CSV кроме user_id и churn
    FEATURES_PATH = next((p for p in FEATURES_PATHS if os.path.exists(p)), None)
    if FEATURES_PATH is None:
        raise FileNotFoundError("CSV с фичами не найден ни по одному из путей.")
    features_df = pd.read_csv(FEATURES_PATH)
    MODEL_FEATURES = [col for col in features_df.columns if col not in ["user_id", "churn"]]

#создаём фиктивный payload
payload = {col: 0 for col in MODEL_FEATURES}

print("Отправляем payload:")
print(payload)

#отправляем запрос на API
try:
    res = requests.post(API_URL, json=payload)
    res.raise_for_status()
except requests.exceptions.RequestException as e:
    print("Ошибка при вызове API:", e)
else:
    try:
        response_json = res.json()
        print("Ответ от API:")
        print(response_json)
    except ValueError:
        print("Не удалось декодировать JSON, ответ API:")
        print(res.text)
