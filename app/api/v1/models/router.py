import pickle

from api.v1.models.schemas import TrainingStatusResponse, TrainModelResponse
from celery.result import AsyncResult
from core.celery_app import app as celery_app
from core.celery_app import train_model_task
from core.schemas import Hyperparameters
from fastapi import APIRouter

router = APIRouter(prefix="/models", tags=["models"])


@router.post("/train", response_model=TrainModelResponse)
async def train_model(hp: Hyperparameters):
    job = train_model_task.delay(pickle.dumps(hp))
    return TrainModelResponse(job_id=job.id)


@router.get("/train/status/{job_id}", response_model=TrainingStatusResponse)
async def get_training_status(job_id: str):
    job_result: AsyncResult = celery_app.AsyncResult(job_id)
    return TrainingStatusResponse(
        status=job_result.status,
        result=pickle.loads(job_result.result) if job_result.successful() else None,
    )


@router.post("/predict")
async def predict():
    return {"message": "predict"}


@router.get("/predict/status/{job_id}")
async def get_prediction_status(job_id: str):
    return {"message": "predict status"}
