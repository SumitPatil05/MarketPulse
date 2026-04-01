from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw_data.csv"


@dataclass(frozen=True)
class SplitData:
    """Container for time-based split outputs."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_random_seed(seed: int = 42) -> None:
    np.random.seed(seed)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Avoid division-by-zero while preserving business meaning."""
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def time_train_test_split(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    target_col: str,
    test_size: int,
) -> SplitData:
    """
    Time-based split (no shuffle) to mimic real forecasting conditions.

    Parameters
    - df: sorted by date
    - test_size: number of final rows to reserve for test
    """
    if test_size <= 0 or test_size >= len(df):
        raise ValueError("test_size must be > 0 and < len(df)")

    X = df.loc[:, list(feature_cols)]
    y = df.loc[:, target_col]

    X_train = X.iloc[:-test_size].copy()
    X_test = X.iloc[-test_size:].copy()
    y_train = y.iloc[:-test_size].copy()
    y_test = y.iloc[-test_size:].copy()
    return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def coerce_datetime(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def latest_row(df: pd.DataFrame, date_col: str = "date") -> pd.Series:
    if df.empty:
        raise ValueError("DataFrame is empty")
    if date_col in df.columns:
        return df.sort_values(date_col).iloc[-1]
    return df.iloc[-1]
