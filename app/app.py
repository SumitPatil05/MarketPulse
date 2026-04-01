from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generation import GenerationConfig, generate_synthetic_data, save_raw_data
from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import build_features
from src.forecasting import forecast_next_days
from src.model import train_linear_regression


DATA_PATH = PROJECT_ROOT / "data" / "raw_data.csv"


def ensure_data_exists() -> None:
    if DATA_PATH.exists():
        return
    df = generate_synthetic_data(GenerationConfig())
    save_raw_data(df, DATA_PATH)


def line_chart(df: pd.DataFrame, x: str, ys: list[str], title: str, y_label: str) -> plt.Figure:
    fig = plt.figure(figsize=(10, 4))
    for y in ys:
        plt.plot(df[x], df[y], label=y, linewidth=2)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="Market-Aware Business Forecasting", layout="wide")
    st.title("Market-Aware Business Forecasting System")
    st.caption("Forecasting brand sales and market share using category dynamics and recent sales history.")

    ensure_data_exists()
    df_raw = load_data(DATA_PATH)
    df_clean, _ = preprocess_data(df_raw)
    df_features = build_features(df_clean)

    feature_cols = [
        "category_sales",
        "category_growth",
        "last_month_sales",
        "last_year_sales",
        "lag_7",
        "lag_30",
    ]

    _, model_result = train_linear_regression(
        df_features=df_features,
        feature_cols=feature_cols,
        target_col="brand_sales",
        test_size=60,
    )

    forecast_res = forecast_next_days(df_features=df_features, model=model_result.model, feature_cols=feature_cols, n_days=30)

    latest = df_clean.sort_values("date").iloc[-1]

    # --- KPI Metrics ---
    st.subheader("1) KPI Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Latest Brand Sales", f"{latest['brand_sales']:,.0f}")
    c2.metric("Latest Market Share", f"{latest['market_share']*100:,.2f}%")
    c3.metric("Latest Category Growth (DoD)", f"{latest['category_growth']*100:,.2f}%")

    # --- Charts ---
    st.subheader("2) Charts")
    left, right = st.columns(2)

    with left:
        fig1 = line_chart(
            df_clean,
            x="date",
            ys=["brand_sales", "category_sales"],
            title="Brand vs Category Sales",
            y_label="Sales",
        )
        st.pyplot(fig1, clear_figure=True)

    with right:
        fig2 = plt.figure(figsize=(10, 4))
        plt.plot(df_clean["date"], df_clean["market_share"], label="market_share", linewidth=2)
        plt.title("Market Share Trend")
        plt.xlabel("Date")
        plt.ylabel("Market Share")
        plt.ylim(0, max(0.4, float(df_clean["market_share"].max()) * 1.1))
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig2, clear_figure=True)

    # Forecast vs actual (using last 60-day test window and next 30-day forecast)
    st.subheader("3) Forecast vs Actual")
    test_dates = df_features.sort_values("date").iloc[-len(model_result.y_true) :]["date"]
    df_test_plot = pd.DataFrame(
        {
            "date": test_dates.values,
            "actual_brand_sales": model_result.y_true,
            "predicted_brand_sales": model_result.y_pred,
        }
    )
    fig3 = line_chart(
        df_test_plot,
        x="date",
        ys=["actual_brand_sales", "predicted_brand_sales"],
        title="Holdout: Actual vs Predicted Brand Sales",
        y_label="Brand Sales",
    )
    st.pyplot(fig3, clear_figure=True)

    # --- Forecast table ---
    st.subheader("4) Next 30 Days Prediction")
    out = forecast_res.forecast_df.copy()
    out["forecast_brand_sales"] = out["forecast_brand_sales"].round(2)
    st.dataframe(out, use_container_width=True)

    with st.expander("Data preview"):
        st.dataframe(df_clean.tail(10), use_container_width=True)


if __name__ == "__main__":
    main()

