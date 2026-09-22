from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import numpy as np
import joblib


app = FastAPI(
    title="Industrial Predictive Maintenance API",
    description="API for predicting machine failure using a Random Forest model.",
    version="1.0.0"
)


# Load saved model artifact
artifact = joblib.load("predictive_maintenance_model.pkl")

model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]


# -----------------------------
# Input Schema Validation
# -----------------------------
class MachineInput(BaseModel):

    type: Literal["L", "M", "H"]

    air_temperature: float = Field(gt=0)
    process_temperature: float = Field(gt=0)

    rotational_speed: int = Field(gt=0)

    torque: float = Field(ge=0)
    tool_wear: int = Field(ge=0)


# Convert Pydantic validation errors to HTTP 400
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
):
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid input data",
            "details": exc.errors()
        }
    )


@app.get("/")
def home():
    return {
        "message": "Industrial Predictive Maintenance API",
        "status": "running",
        "model": "RandomForestClassifier",
        "threshold": threshold
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.post("/predict")
def predict(data: MachineInput):

    # Feature Engineering
    temperature_difference = (
        data.process_temperature
        - data.air_temperature
    )

    power = (
        data.torque
        * data.rotational_speed
        * 2
        * np.pi
        / 60
    )

    torque_wear_interaction = (
        data.torque
        * data.tool_wear
    )

    # One-hot encoding for machine type
    type_h = int(data.type == "H")
    type_l = int(data.type == "L")
    type_m = int(data.type == "M")

    # Create model input using EXACT training feature names
    input_data = pd.DataFrame([{
        "Air temperature [K]": data.air_temperature,
        "Process temperature [K]": data.process_temperature,
        "Rotational speed [rpm]": data.rotational_speed,
        "Torque [Nm]": data.torque,
        "Tool wear [min]": data.tool_wear,
        "Temperature Difference [K]": temperature_difference,
        "Power [W]": power,
        "Torque_Wear_Interaction": torque_wear_interaction,
        "Type_H": type_h,
        "Type_L": type_l,
        "Type_M": type_m
    }])

    # Guarantee same column order used during training
    input_data = input_data[features]

    # Failure probability
    failure_probability = model.predict_proba(
        input_data
    )[0, 1]

    # Apply threshold selected in Task 9
    prediction = int(
        failure_probability >= threshold
    )

    return {
        "prediction": prediction,
        "prediction_label": (
            "Failure"
            if prediction == 1
            else "No Failure"
        ),
        "failure_probability": round(
            float(failure_probability), 4
        ),
        "threshold": threshold
    }