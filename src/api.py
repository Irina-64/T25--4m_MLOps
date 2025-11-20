from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel
import os

# Загружаем модель и preprocessing artifacts
model = joblib.load("models/best_model.joblib")
scaler = joblib.load("models/scaler.joblib")
feature_names = joblib.load("models/feature_names.joblib")

app = FastAPI(title="Flight Delay Prediction API", 
              description="API для предсказания направления изменения курса валют",
              version="1.0")

# Определяем структуру входных данных
class PredictionInput(BaseModel):
    USD_RUB: float
    EUR_RUB: float
    GBP_RUB: float
    day_of_week: int
    is_weekend: int
    USD_RUB_lag_1: float
    USD_RUB_lag_2: float
    USD_RUB_lag_3: float
    USD_RUB_lag_5: float
    USD_RUB_lag_7: float
    EUR_RUB_lag_1: float
    EUR_RUB_lag_2: float
    EUR_RUB_lag_3: float
    EUR_RUB_lag_5: float
    EUR_RUB_lag_7: float
    GBP_RUB_lag_1: float
    GBP_RUB_lag_2: float
    GBP_RUB_lag_3: float
    GBP_RUB_lag_5: float
    GBP_RUB_lag_7: float
    USD_RUB_MA_3: float
    USD_RUB_MA_5: float
    USD_RUB_MA_7: float
    EUR_RUB_MA_3: float
    EUR_RUB_MA_5: float
    EUR_RUB_MA_7: float
    GBP_RUB_MA_3: float
    GBP_RUB_MA_5: float
    GBP_RUB_MA_7: float
    USD_RUB_change_1: float
    USD_RUB_change_3: float

@app.get("/")
def read_root():
    return {"message": "Flight Delay Prediction API", "version": "1.0"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict(input_data: PredictionInput):
    """
    Предсказание направления изменения курса USD/RUB
    Returns: вероятность роста курса (класс 1)
    """
    # Конвертируем входные данные в DataFrame
    input_dict = input_data.dict()
    df = pd.DataFrame([input_dict])
    
    # Убеждаемся что порядок признаков правильный
    df = df[feature_names]
    
    # Масштабируем признаки
    scaled_data = scaler.transform(df)
    
    # Предсказание
    prediction_proba = model.predict_proba(scaled_data)
    
    # Вероятность класса 1 (рост курса)
    delay_prob = float(prediction_proba[0, 1])
    
    return {
        "delay_prob": delay_prob,
        "prediction": "rise" if delay_prob > 0.5 else "fall",
        "confidence": delay_prob if delay_prob > 0.5 else 1 - delay_prob
    }

# Для тестирования без Docker
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
