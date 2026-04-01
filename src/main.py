from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_generation import GenerationConfig, generate_synthetic_data, save_raw_data
from .data_preprocessing import load_data, preprocess_data
from .evaluate import print_metrics, regression_metrics
from .feature_engineering import build_features
from .forecasting import forecast_next_days
from .model import train_linear_regression
from .visualization import (
    plot_actual_vs_predicted,
    plot_category_vs_brand_trend,
    plot_market_share_trend,
)


def run_pipeline(project_root: Path) -> None:
    data_path = project_root / "data" / "raw_data.csv"

    # 1) Generate data (idempotent: overwrite for reproducibility)
    df_raw = generate_synthetic_data(GenerationConfig())
    save_raw_data(df_raw, data_path)

    # 2) Preprocess
    df_loaded = load_data(data_path)
    df_clean, df_missing = preprocess_data(df_loaded)
    if len(df_missing) > 0:
        print(f"Note: {len(df_missing)} rows had missing values (kept for inspection).")

    # 3) Feature engineering
    df_features = build_features(df_clean)

    # 4) Model building
    feature_cols = [
        "category_sales",
        "category_growth",
        "last_month_sales",
        "last_year_sales",
        "lag_7",
        "lag_30",
    ]
    split, result = train_linear_regression(
        df_features=df_features,
        feature_cols=feature_cols,
        target_col="brand_sales",
        test_size=60,
    )

    # 5) Evaluation
    metrics = regression_metrics(result.y_true, result.y_pred)
    print_metrics(metrics)

    # 6) Visualization
    test_dates = df_features.sort_values("date").iloc[-len(result.y_true) :]["date"]
    plot_actual_vs_predicted(test_dates, result.y_true, result.y_pred)
    plot_category_vs_brand_trend(df_clean)
    plot_market_share_trend(df_clean)

    # 7) Forecasting (next 30 days)
    forecast = forecast_next_days(df_features=df_features, model=result.model, feature_cols=feature_cols, n_days=30)
    print("\nNext 30 Days Forecast (Brand Sales)")
    print("-" * 34)
    out = forecast.forecast_df.copy()
    out["forecast_brand_sales"] = out["forecast_brand_sales"].round(2)
    print(out.to_string(index=False))


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    run_pipeline(project_root)


if __name__ == "__main__":
    main()

