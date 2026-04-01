from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    # Version-safe RMSE: some sklearn builds do not support `squared=False`.
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"MAE": mae, "RMSE": rmse}


def print_metrics(metrics: dict) -> None:
    print("Model Evaluation Metrics")
    print("-" * 26)
    for k, v in metrics.items():
        print(f"{k}: {v:,.2f}")
