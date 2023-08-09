from asyncio import sleep

import pytest
from httpx import AsyncClient

from app.core.settings import settings
from app.main import app

BASE_URL = f"http://{settings.APP_HOST}:{settings.APP_PORT}/api/v1"
POLLING_INTERVAL = 5
WAIT_TIME = 60


async def poll_train_job(
    ac: AsyncClient,
    job_id: str,
    polling_interval: int = POLLING_INTERVAL,
    wait_time: int = WAIT_TIME,
) -> bool:
    for _ in range(WAIT_TIME // POLLING_INTERVAL):
        response = await ac.get(f"/api/v1/models/train/status/{job_id}")
        assert response.status_code == 200

        match response.json()["status"]:
            case "SUCCESS":
                return True
            case "FAILURE":
                return False
            case _:
                sleep(polling_interval)

    raise AssertionError("Timeout")


@pytest.mark.asyncio
async def test_train_model():
    async with AsyncClient(app=app, base_url=app) as ac:
        response = await ac.post(
            "/api/v1/models/train",
            json={
                "label_column": "expected_loss",
                "epochs": 10,
                "learning_rate": 0.01,
                "batch_size": 32,
                "sgd_weight_decay": 0.01,
                "test_samples_count": 100,
            },
        )
        assert response.status_code == 200

        job_id = response.json()["job_id"]
        assert await poll_train_job(ac, job_id) is True
