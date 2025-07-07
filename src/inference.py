import numpy as np
import joblib
import io
import pandas as pd
import os
from typing import List, Union
from .config import MODEL_LOGISTIC,MODEL_KNN,MODEL_Random,MODEL_SVM,SCALER
from .schemas import PredictionResponse,ModelInputBatch

# Load the scaler
SRC_FOLDER_PATH=os.path.dirname(os.path.abspath(__file__))
SCALER = joblib.load(os.path.join(SRC_FOLDER_PATH,'assets','scaler.pkl')) 

# Map model names to model objects
MODELS = {
    "logistic": MODEL_LOGISTIC,
    "knn": MODEL_KNN,
    "random_forest": MODEL_Random,
    "svm": MODEL_SVM
}

# Exact feature order and names used during training
FEATURES = [
    "perimeter_mean",
    "area_mean",
    "concavity_mean",
    "concave points_mean",  # ⚠️ space, not underscore
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "concave points_worst",  # ⚠️ space
    "radius_texture_interaction"
]

def preprocess_input(input_batch: ModelInputBatch) -> np.ndarray:
    df = pd.DataFrame([item.dict() for item in input_batch.inputs])
    df.columns = [col.replace('_', ' ') if 'concave points' in col else col]  # Handle the space issue if needed
    df = df[FEATURES]  # Reorder and select only used features
    return SCALER.transform(df)


def predict(model_name:str ,input_batch:ModelInputBatch)->PredictionResponse:
    """Make predictions using the specified model and input data."""
    # Get the model object from the map
    model=MODELS.get(model_name.lower())
    if model is None:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(MODELS.keys())}")
    
    # Preprocess input
    X_scaled = preprocess_input(input_batch)

    # Predict
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled) if hasattr(model, 'predict_proba') else None

    # Build response
    predictions = []
    for idx, pred in enumerate(y_pred):
        pred_dict = {
            "input_index": idx,
            "predicted_class": int(pred)
        }
        if y_proba is not None:
            pred_dict["probabilities"] = y_proba[idx].tolist()
        
        predictions.append(pred_dict)
    
    return PredictionResponse(predictions=predictions)