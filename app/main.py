import os
import joblib
import pandas as pd
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# 1. Initialize FastAPI
app = FastAPI(title="Pasture.io Yield Prediction API")

# 2. Configuration & Model Loading
# In production, we load the 'Production' version from the Registry
MODEL_NAME = "Yield_Model_XGB"
MODEL_STAGE = "Production" 
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

# Load the Preprocessor (Scaler + Encoder) and the Model
try:
    # Set tracking URI to DagsHub
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    
    print("Fetching model from DagsHub...")
    model = mlflow.pyfunc.load_model(MODEL_URI)
    
    print("Loading local preprocessor...")
    preprocessor = joblib.load('models/preprocessor.pkl')
    print("System Ready.")
except Exception as e:
    print(f"Error loading production artifacts: {e}")

class FarmData(BaseModel):
    Temperature: float
    Humidity: float
    Rainfall: float
    Soil_Type: str
    pH: float
    EC: float
    OC: float
    N: float
    P: float
    K: float
    Ca: float
    Mg: float
    S: float
    Zn: float
    Fe: float
    Cu: float
    Mn: float
    B: float
    Mo: float
    CEC: float
    Sand: float
    Silt: float
    Clay: float
    Bulk_Density: float
    Water_Holding_Capacity: float
    Slope: float
    Aspect: float
    Elevation: float
    Solar_Radiation: float
    Wind_Speed: float
    NDVI: float
    EVI: float
    LAI: float
    Chlorophyll: float
    GDD: float
    Crop_Type: str
    Growth_Stage: str
    Irrigation_Frequency: int
    Fertilizer_Type: str
    Pesticide_Usage: str
    Region: str
    Season: str

@app.get("/health")
def health():
    return {"status": "online", "model": MODEL_NAME, "stage": MODEL_STAGE}

@app.post("/predict")
def predict(data: FarmData):
    try:
        input_df = pd.DataFrame([data.dict()])
        
        # Apply the SAME preprocessing used in training
        processed_data = preprocessor.transform(input_df)
        
        # Inference
        prediction = model.predict(processed_data)
        
        return {
            "prediction_unit": "Yield (Value)",
            "predicted_yield": float(prediction[0]),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))