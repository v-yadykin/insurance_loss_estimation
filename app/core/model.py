from time import process_time

import numpy as np
import torch
from core.schemas import Hyperparameters, ModelEvaluationResult
from loguru import logger
from numba import jit
from pandas import DataFrame
from torch import Tensor, nn
from torch.nn.functional import l1_loss
from torch.utils.data import DataLoader, Dataset


class RegressionDataset(Dataset):
    def __init__(self, df: DataFrame, target_column: str):
        self.n = df.shape[0]

        self.y: Tensor = torch.from_numpy(
            df[target_column].values.astype(np.float64).reshape(-1, 1)
        )

        x_columns = list(filter(lambda name: name != target_column, df.columns.values))
        self.x: Tensor = torch.from_numpy(df[x_columns].values.astype(np.float64))

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class LinearRegression(nn.Module):
    def __init__(self, input_dim: int):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1, dtype=torch.float64)

    def forward(self, x: Tensor):
        return self.linear(x)


def train(df: DataFrame, hp: Hyperparameters) -> LinearRegression:
    dataset = RegressionDataset(df, hp.label_column)
    dataloader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True)
    model = LinearRegression(df.shape[1] - 1)
    criterion = nn.L1Loss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=hp.learning_rate, weight_decay=hp.sgd_weight_decay
    )

    for epoch in range(hp.epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        logger.info(f"{epoch=} loss={loss.item()}")

    return model


def evaluate(
    model: nn.Module, df: DataFrame, target_column: str, loss=lambda x, y: l1_loss(x, y).item()
) -> ModelEvaluationResult:
    start_time = process_time()

    labels: Tensor = torch.from_numpy(df[target_column].values.astype(np.float64).reshape(-1, 1))

    x_columns = list(filter(lambda name: name != target_column, df.columns.values))
    inputs: Tensor = torch.from_numpy(df[x_columns].values.astype(np.float64))

    with torch.no_grad():
        predicted = model(inputs)

    accuracy = loss(predicted, labels)
    calc_time = process_time() - start_time

    return ModelEvaluationResult(accuracy=accuracy, time=calc_time)


@jit(nopython=True, parallel=True)
def numba_abs_error(predicted: np.ndarray, labels: np.ndarray):
    return np.abs(predicted - labels).mean()


def numba_loss(predicted: Tensor, labels: Tensor):
    return numba_abs_error(predicted.numpy(force=True), labels.numpy(force=True))
