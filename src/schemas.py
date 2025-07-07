from pydantic import BaseModel
from typing import List

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    perimeter_mean: float
    area_mean: float
    concavity_mean: float
    concave_points_mean: float
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    concave_points_worst: float
    radius_texture_interaction: float


class ModelInputBatch(BaseModel):
    inputs: List[PredictionResponse]    