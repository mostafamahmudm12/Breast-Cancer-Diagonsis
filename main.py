from fastapi import FastAPI, Depends, HTTPException,UploadFile,File,BackgroundTasks
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union, Tuple, Callable, TypeVar
import os 
import logging
from src.schemas import ModelInputBatch, PredictionResponse
from src.inference import predict
import uvicorn


# Custom Modules
from src.config import APP_NAME, VERSION, API_SECRET_KEY


# Basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

app = FastAPI(title=APP_NAME, version=VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
logging.info(f"App started: {APP_NAME} v{VERSION}")

# API key verification
api_key_header = APIKeyHeader(name='X-API-Key')
async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_SECRET_KEY:
        logging.warning("Unauthorized API access attempt")
        raise HTTPException(status_code=403, detail="You are not authorized to use this API")
    return api_key

@app.get('/', tags=['check'])
async def home(api_key: str=Depends(verify_api_key)):
    return {
        "app_name": APP_NAME,
        "version": VERSION,
        "status": "up & running"
    }



@app.post("/predict/{model_name}", response_model=PredictionResponse)
def predict_route(model_name: str, input_data: ModelInputBatch):
    try:
        return predict(model_name, input_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)



