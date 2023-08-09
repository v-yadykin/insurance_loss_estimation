import pickle

from celery import Celery
from core.model import evaluate, numba_loss, train
from core.schemas import Hyperparameters, ModelTrainingResult
from core.settings import settings
from pandas import read_parquet

app = Celery(
    main=f"{settings.APP_NAME}_worker",
    backend="db+postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}".format(
        user=settings.DB_USERNAME,
        pwd=settings.DB_PASSWORD,
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        db=settings.DB_NAME,
    ),
    broker_url="amqp://{user}:{pwd}@{host}:{port}".format(
        user=settings.MQ_USERNAME,
        pwd=settings.MQ_PASSWORD,
        host=settings.MQ_HOST,
        port=settings.MQ_PORT,
    ),
)


@app.task
def train_model_task(hyperparametres: bytes) -> bytes:
    train_data = read_parquet(settings.DATA_PATH, engine="fastparquet")
    hp: Hyperparameters = pickle.loads(hyperparametres)

    for col in train_data.select_dtypes(include=["object"]).columns:
        train_data[col] = train_data[col].factorize()[0]

    model = train(train_data, hp)

    test_samples = train_data.sample(n=hp.test_samples_count)

    return pickle.dumps(
        ModelTrainingResult(
            single_sample=evaluate(model, test_samples.sample(1), hp.label_column),
            test_samples=evaluate(model, test_samples, hp.label_column),
            test_samples_numba=evaluate(model, test_samples, hp.label_column, numba_loss),
        )
    )


if __name__ == "__main__":
    app.start()
