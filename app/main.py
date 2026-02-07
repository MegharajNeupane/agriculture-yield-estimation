import os
import joblib
import pandas as pd
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()
artifacts = {}



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        model_uri = f"models:/Yield_Model_XGB@champion"
        artifacts["model"] = mlflow.pyfunc.load_model(model_uri)
        artifacts["preprocessor"] = joblib.load('models/preprocessor.pkl')
        print("✅ Production Ready.")
    except Exception as e:
        print(f"❌ Error: {e}")
    yield
    artifacts.clear()

app = FastAPI(title="Yield API", lifespan=lifespan)


@app.get("/")
def read_root():
    return {"message": "Yield Prediction API is Online. Visit /docs for documentation."}
class FarmData(BaseModel):
    Temperature: float = Field(..., json_schema_extra={"example": 25.0})
    Humidity: float
    Rainfall: float
    Soil_Type: str
    pH: float = Field(..., ge=0, le=14)
    EC: float; OC: float; N: float; P: float; K: float
    Ca: float; Mg: float; S: float; Zn: float; Fe: float
    Cu: float; Mn: float; B: float; Mo: float; CEC: float
    Sand: float; Silt: float; Clay: float
    Bulk_Density: float; Water_Holding_Capacity: float
    Slope: float; Aspect: float; Elevation: float
    Solar_Radiation: float; Wind_Speed: float
    NDVI: float = Field(..., ge=-1, le=1)
    EVI: float; LAI: float; Chlorophyll: float; GDD: float
    Crop_Type: str; Growth_Stage: str; Irrigation_Frequency: int
    Fertilizer_Type: str; Pesticide_Usage: str; Region: str; Season: str

@app.post("/predict")
async def predict(data: FarmData):
    input_df = pd.DataFrame([data.model_dump()])
    # Ensure consistency with training
    input_df = input_df.drop(columns=['Planting_Date', 'Harvest_Date', 'Year'], errors='ignore')
    
    processed = artifacts["preprocessor"].transform(input_df)
    pred = artifacts["model"].predict(processed)
    return {"predicted_yield": float(pred[0])}