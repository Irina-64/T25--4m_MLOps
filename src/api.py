import json

import joblib
import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .tools.decoder import apply_encoders_and_scaler
from .tools.json_db import Db

MODEL_PATH = "models/final_xgboost_model.pkl"
ENCODERS_PATH = "models/encoders.pkl"
SCALER_PATH = "models/scaler.pkl"
LABELS_PATH = "src/files/labels.json"
DATA_PATH = "src/files/d.json"
PARAMS_PATH = "src/files/parameters.json"

db = Db(LABELS_PATH, DATA_PATH)

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Ошибка загрузки модели: {e}")

try:
    encoders = joblib.load(ENCODERS_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Ошибка загрузки энкодеров или скейлера: {e}")

with open(PARAMS_PATH, "r", encoding="utf-8") as f:
    params = json.load(f)

app = FastAPI(title="Car Price Prediction API", version="1.0")
router = APIRouter(prefix="/api/v1")


class CarBase(BaseModel):
    mark: str


class CarModel(CarBase):
    model: str


class CarGeneration(CarModel):
    super_gen_name: str


class PredictRequest(BaseModel):
    mark: str
    model: str
    super_gen_name: str
    body_type_type: str
    transmission: str
    engine: str
    year: int
    mileage: float
    color: str
    owners: str
    region: str
    gear_type: str
    steering_wheel: str
    complectation: str
    power: float
    displacement: float


@router.get("/params")
async def get_params():
    return JSONResponse(params)


@router.get("/marks")
async def get_marks():
    return JSONResponse(db.get_marks())


@router.get("/steering_wheels")
async def get_steering_wheels():
    return JSONResponse(db.get_steering_wheel())


@router.get("/owners")
async def get_owners():
    return JSONResponse(db.get_owners())


@router.get("/regions")
async def get_regions():
    return JSONResponse(db.get_regions())


@router.get("/gear_types")
async def get_gear_type():
    return JSONResponse(db.get_gear_type())


@router.get("/colors")
async def get_colors():
    return JSONResponse(db.get_colors())


@router.post("/models")
async def get_models(data: CarBase):
    return JSONResponse(db.get_models_car(data.mark))


@router.post("/gens")
async def get_gens(data: CarModel):
    return JSONResponse(db.get_super_gen_names_car(data.mark, data.model))


@router.post("/bodies")
async def get_bodies(data: CarGeneration):
    return JSONResponse(db.get_body_types_car(data.mark, data.model, data.super_gen_name))


@router.post("/complectations")
async def get_complectations(data: CarGeneration):
    return JSONResponse(db.get_complectations_car(data.mark, data.model, data.super_gen_name))


@router.post("/transmission")
async def get_transmission(data: CarGeneration):
    return JSONResponse(db.get_transmissions_car(data.mark, data.model, data.super_gen_name))


@router.post("/engines")
async def get_engines(data: CarGeneration):
    return JSONResponse(db.get_engines_car(data.mark, data.model, data.super_gen_name))


@router.post("/years")
async def get_years(data: CarGeneration):
    return JSONResponse(db.get_years_car(data.mark, data.model, data.super_gen_name))


@router.post("/predict")
async def predict(data: PredictRequest):
    try:
        processed_df = apply_encoders_and_scaler(data.dict(), encoders=encoders, scaler=scaler)

        X = processed_df.values
        pred_scaled = model.predict(X)

        if hasattr(scaler, "mean_") and "price" in scaler.feature_names_in_:
            temp = np.zeros((1, len(scaler.feature_names_in_)))
            price_idx = list(scaler.feature_names_in_).index("price")
            temp[0, price_idx] = pred_scaled[0]

            temp_real = scaler.inverse_transform(temp)
            price_real = temp_real[0, price_idx]
        else:
            price_real = float(pred_scaled[0])

        return JSONResponse({"predicted": float(price_real)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при предсказании: {e}")


app.include_router(router)
