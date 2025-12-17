from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os
import numpy as np


app = FastAPI()

MODEL_DIR = "models"
MODEL_CACHE = {}


class PredictRequest(BaseModel):
    branch_id:str
    item_codes:list[str]
    n:int

def load_model(branch_id:str):

    "if model is alredy present in then return "
    if branch_id in MODEL_CACHE:
        return MODEL_CACHE[branch_id]

    model_path = os.path.join(MODEL_DIR, f"{branch_id}_sales_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model Not found")


    with open(model_path, "rb") as f:
        model = pickle.load(f)

    "caching the model to load faster next"
    MODEL_CACHE[branch_id] = model
    return model



@app.post("/predict")
def predict(req: PredictRequest):
    try:
        model = load_model(req.branch_id)

        "Predictions for n days"
        forecast = model.predict(req.n)
        forecast["prediction"] = np.expm1(forecast["LGBMRegressor"])
        forecast["item_code"] = forecast["id"].str.split("_").str[-1]

        "Taking only required items predictions"
        forecast = forecast[forecast["item_code"].isin(req.item_codes)]


        result = (
            forecast.groupby("item_code")["prediction"].sum().round(2).to_dict()
        )

        return result

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


