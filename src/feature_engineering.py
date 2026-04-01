from __future__ import annotations

import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month"] = out["date"].dt.month.astype(int)
    out["day_of_week"] = out["date"].dt.dayofweek.astype(int)  # Monday=0
    return out


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lag/rolling features based on brand_sales.

    Business logic:
    - Recent history is often predictive (promo pull-forward, retail replenishment, etc.).
    """
    out = df.copy()
    out["lag_7"] = out["brand_sales"].shift(7)
    out["lag_30"] = out["brand_sales"].shift(30)
    out["rolling_mean_7"] = out["brand_sales"].rolling(window=7, min_periods=7).mean()
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_time_features(df)
    out = add_lag_features(out)

    # Drop NA rows created by lag/rolling features (initial window).
    out = out.dropna().reset_index(drop=True)
    return out
