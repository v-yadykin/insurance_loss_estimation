from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App basic settings
    API_V1_STR: str = "/api/v1"
    APP_NAME: str = "loss_estimation"
    APP_HOST: str
    APP_PORT: int

    # Database
    DB_USERNAME: str
    DB_PASSWORD: str
    DB_HOST: str
    DB_PORT: int
    DB_NAME: str

    # Message queue
    MQ_USERNAME: str
    MQ_PASSWORD: str
    MQ_NAME: str
    MQ_HOST: str
    MQ_PORT: int
    MQ_UI_PORT: str

    # Model training
    DATA_PATH: Path


settings = Settings()  # type: ignore
