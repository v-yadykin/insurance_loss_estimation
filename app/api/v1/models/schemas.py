from typing import Optional

from core.schemas import ModelTrainingResult
from pydantic import BaseModel


class TrainModelResponse(BaseModel):
    status_code: int = 200
    job_id: str
    message: str = "Model training started"


class TrainingStatusResponse(BaseModel):
    status_code: int = 200
    status: str
    result: Optional[ModelTrainingResult] = None
