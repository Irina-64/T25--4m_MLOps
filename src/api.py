from fastapi import FastAPI, UploadFile, File
import torch
import numpy as np
from pathlib import Path
import pandas as pd
import os
import tempfile
from src.inference import prepare_sequences, LSTMChurnModel, HIDDEN_SIZE, main

app = FastAPI(title="Churn Prediction Service")

# Загрузка модели и скейлера 
checkpoint = torch.load("model.pt", map_location=torch.device("cpu"))
model = LSTMChurnModel(input_size=2, hidden_size=HIDDEN_SIZE)
model.load_state_dict(checkpoint["model"])
model.eval()

# Воссоздаем скейлер
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.mean_ = checkpoint["scaler_mean"]
scaler.scale_ = checkpoint["scaler_scale"]

@app.post("/predict")
async def predict(payload: dict):
    try:
        # Преобразуем входные данные в DataFrame
        df = pd.DataFrame(payload["transactions"])
        df["user_id"] = 0  # временный id для одного пользователя

        # Используем функцию prepare_sequences из inference.py
        user_ids, X = prepare_sequences(df, scaler)
        
        # Получение предсказания
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            output = model(X_tensor)
            churn_probability = float(output.item())
        
        return {
            "churn_probability": churn_probability
        }
    
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    try:
        content = await file.read()

        # Создаём временные файлы для входа и вывода
        tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp_in.write(content)
        tmp_in.close()
        tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp_out.close()

        try:
            # Запускаем основную функцию inference.main(input_path, output_path)
            result = main(tmp_in.name, tmp_out.name)

            # Если main вернул DataFrame — используем его
            if hasattr(result, "to_dict"):
                df = result
            # Иначе, если main сохранил файл — читаем его
            elif os.path.exists(tmp_out.name) and os.path.getsize(tmp_out.name) > 0:
                df = pd.read_csv(tmp_out.name)
            # Если main вернул список/словарь — возвращаем как есть
            elif isinstance(result, (list, dict)):
                return result
            else:
                # Попытка преобразовать возвращённое значение в DataFrame
                df = pd.DataFrame(result)

            return df.to_dict(orient="records")

        finally:
            # Удаляем временные файлы
            try:
                os.remove(tmp_in.name)
            except OSError:
                pass
            try:
                os.remove(tmp_out.name)
            except OSError:
                pass

    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"message": "Churn Prediction API is running"}

