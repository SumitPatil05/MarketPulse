from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from .utils import coerce_datetime, validate_columns


REQUIRED_COLUMNS = [
    "date",
    "brand_sales",
    "category_sales",
    "category_growth",
    "last_month_sales",
    "last_year_sales",
    "market_share",
]


def load_data(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Basic cleaning suitable for a forecasting pipeline.

    Returns:
    - df_clean: cleaned + sorted
    - df_missing: rows with any missing values (useful for sanity checks)
    """
    validate_columns(df, REQUIRED_COLUMNS)

    out = coerce_datetime(df, "date")
    out = out.sort_values("date").reset_index(drop=True)

    # For this project we keep it simple: forward-fill lagged fields and growth,
    # and drop rows where the date is invalid.
    out = out.dropna(subset=["date"])

    missing_mask = out.isna().any(axis=1)
    df_missing = out.loc[missing_mask].copy()

    fill_cols = ["last_month_sales", "category_growth", "market_share", "last_year_sales"]
    out[fill_cols] = out[fill_cols].ffill()

    # Sales should never be missing; if they are, we drop them (rare in synthetic data).
    out = out.dropna(subset=["brand_sales", "category_sales"])

    return out, df_missing


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    df = load_data(project_root / "data" / "raw_data.csv")
    df_clean, df_missing = preprocess_data(df)
    print(f"Rows: {len(df_clean)} | Missing rows captured: {len(df_missing)}")


if __name__ == "__main__":
    main()
