from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os
import numpy as np

app = FastAPI()

MODEL_DIR = "models"
MODEL_CACHE = {}


class PredictRequest(BaseModel):
    branch_id: str
    item_codes: list[str]
    n: int
    mode: str | None = "SUM"


def load_model(branch_id: str):
    if branch_id in MODEL_CACHE:
        return MODEL_CACHE[branch_id]

    model_path = os.path.join(MODEL_DIR, f"{branch_id}_sales_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    MODEL_CACHE[branch_id] = model
    return model


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        if not req.branch_id or not req.item_codes or req.n <= 0:
            raise HTTPException(status_code=400, detail="Invalid input")

        model = load_model(req.branch_id)

        forecast = model.predict(req.n)

        forecast["prediction"] = np.expm1(forecast["LGBMRegressor"]).clip(lower=0)
        forecast["item_code"] = forecast["id"].str.split("_").str[-1]

        forecast = forecast[forecast["item_code"].isin(req.item_codes)]

        result = {}

        if req.mode == "DAILY":
            for item_code, grp in forecast.groupby("item_code"):
                result[item_code] = (
                    grp.sort_values("ds")["prediction"]
                    .round(3)
                    .tolist()
                )
        else:  
            result = (
                forecast.groupby("item_code")["prediction"]
                .sum()
                .round(2)
                .to_dict()
            )

        return result

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
