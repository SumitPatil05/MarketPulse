from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class ForecastResult:
    forecast_df: pd.DataFrame


def forecast_next_days(
    df_features: pd.DataFrame,
    model: LinearRegression,
    feature_cols: List[str],
    n_days: int = 30,
) -> ForecastResult:
    """
    Forecast next n_days using last available values as inputs.

    Practical simplification (common in portfolio projects):
    - We hold category_sales and category_growth at their last observed values.
    - last_year_sales is held constant at its last observed value.
    - Lag features are updated iteratively using the model's own predictions (recursive forecast).
    """
    if df_features.empty:
        raise ValueError("df_features is empty")

    history = df_features.sort_values("date").copy()
    last_date = history["date"].iloc[-1]

    last_category_sales = float(history["category_sales"].iloc[-1])
    last_category_growth = float(history["category_growth"].iloc[-1])
    last_last_year_sales = float(history["last_year_sales"].iloc[-1])

    # Keep a working list of recent brand_sales values to update lags.
    recent_brand_sales = list(history["brand_sales"].iloc[-60:].astype(float).values)

    rows = []
    for i in range(1, n_days + 1):
        date_i = last_date + pd.Timedelta(days=i)

        # Derive lag features from most recent actual/predicted values
        lag_7 = recent_brand_sales[-7]
        lag_30 = recent_brand_sales[-30]

        last_month_sales = lag_30  # aligns with the "lag 30" business definition
        x = pd.DataFrame(
            [
                {
                    "category_sales": last_category_sales,
                    "category_growth": last_category_growth,
                    "last_month_sales": last_month_sales,
                    "last_year_sales": last_last_year_sales,
                    "lag_7": lag_7,
                    "lag_30": lag_30,
                }
            ]
        )
        x = x.loc[:, feature_cols]

        pred = float(model.predict(x)[0])
        pred = max(pred, 0.0)

        recent_brand_sales.append(pred)
        rows.append({"date": date_i, "forecast_brand_sales": pred})

    forecast_df = pd.DataFrame(rows)
    return ForecastResult(forecast_df=forecast_df)

