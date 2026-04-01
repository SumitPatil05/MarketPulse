from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_actual_vs_predicted(
    dates: pd.Series,
    y_true,
    y_pred,
    output_path: Optional[Path] = None,
) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(dates, y_true, label="Actual", linewidth=2)
    plt.plot(dates, y_pred, label="Predicted", linewidth=2)
    plt.title("Actual vs Predicted Brand Sales")
    plt.xlabel("Date")
    plt.ylabel("Brand Sales")
    plt.legend()
    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
    plt.show()


def plot_category_vs_brand_trend(df: pd.DataFrame, output_path: Optional[Path] = None) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(df["date"], df["category_sales"], label="Category Sales", linewidth=2)
    plt.plot(df["date"], df["brand_sales"], label="Brand Sales", linewidth=2)
    plt.title("Category vs Brand Sales Trend")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
    plt.show()


def plot_market_share_trend(df: pd.DataFrame, output_path: Optional[Path] = None) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(df["date"], df["market_share"], label="Market Share", linewidth=2)
    plt.title("Market Share Trend (Brand / Category)")
    plt.xlabel("Date")
    plt.ylabel("Market Share")
    plt.ylim(0, max(0.4, float(df["market_share"].max()) * 1.1))
    plt.legend()
    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
    plt.show()

