from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .utils import SplitData, time_train_test_split


@dataclass(frozen=True)
class ModelResult:
    model: LinearRegression
    y_pred: np.ndarray
    y_true: np.ndarray
    X_test: pd.DataFrame


def train_linear_regression(
    df_features: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "brand_sales",
    test_size: int = 60,
) -> Tuple[SplitData, ModelResult]:
    """
    Train a linear regression model using a strict time split.

    This mimics a real business setting: we train on the past and evaluate on the most recent period.
    """
    split = time_train_test_split(df_features, feature_cols, target_col, test_size=test_size)

    model = LinearRegression()
    model.fit(split.X_train, split.y_train)

    y_pred = model.predict(split.X_test)
    y_true = split.y_test.values

    return split, ModelResult(model=model, y_pred=y_pred, y_true=y_true, X_test=split.X_test)
