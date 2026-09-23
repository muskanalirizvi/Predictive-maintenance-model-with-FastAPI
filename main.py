from fastapi import FastAPI
from pydantic import BaseModel
import joblib, json, numpy as np, pandas as pd

app = FastAPI(title="Predictive Maintenance API")
model = joblib.load("model.pkl")
config = json.load(open("config.json"))

class Machine(BaseModel):
    type: str            # "L", "M", "H"
    air_temp: float
    process_temp: float
    rpm: float
    torque: float
    tool_wear: float

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/predict")
def predict(m: Machine):
    d = m.model_dump()
    d["type"] = {"L": 0, "M": 1, "H": 2}[d["type"].upper()]
    d["temp_diff"] = d["process_temp"] - d["air_temp"]
    d["power"] = d["torque"] * d["rpm"] * 2 * np.pi / 60
    d["strain"] = d["tool_wear"] * d["torque"]

    X = pd.DataFrame([d])[config["features"]]
    prob = float(model.predict_proba(X)[0][1])
    return {"failure": int(prob >= config["threshold"]), "probability": round(prob, 3)}