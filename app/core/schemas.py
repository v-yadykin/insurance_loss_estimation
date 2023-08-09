from pydantic import BaseModel


class Hyperparameters(BaseModel):
    label_column: str = "expected_loss"
    epochs: int = 10
    learning_rate: float = 0.01
    batch_size: int = 32
    sgd_weight_decay: float = 0.01
    test_samples_count: int = 100


class ModelEvaluationResult(BaseModel):
    accuracy: float
    time: float


class ModelTrainingResult(BaseModel):
    single_sample: ModelEvaluationResult
    test_samples: ModelEvaluationResult
    test_samples_numba: ModelEvaluationResult
