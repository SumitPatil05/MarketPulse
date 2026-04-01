from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import ensure_dir, safe_divide, set_random_seed


@dataclass(frozen=True)
class GenerationConfig:
    """Business-inspired knobs to shape the synthetic market and brand dynamics."""

    n_days: int = 365
    start_date: str = "2025-01-01"
    seed: int = 42
    base_category_sales: float = 120_000.0
    daily_category_trend: float = 120.0  # steady category expansion
    weekly_seasonality_strength: float = 0.08  # amplitude as % of baseline
    category_noise_std: float = 3_500.0
    base_brand_share: float = 0.18
    share_sensitivity_to_growth: float = 0.25  # brand share slightly improves when category is growing
    share_noise_std: float = 0.012
    min_share: float = 0.08
    max_share: float = 0.35


def _weekly_seasonality(day_index: np.ndarray) -> np.ndarray:
    """Smooth weekly pattern (e.g., weekends/weekday effects)."""
    return np.sin(2 * np.pi * day_index / 7.0)


def generate_synthetic_data(config: GenerationConfig) -> pd.DataFrame:
    """
    Create a realistic daily dataset for a brand within a growing category.

    Business logic:
    - Category sales grow gradually, with weekly seasonality + noise.
    - Brand sales are a (noisy) share of category sales, with mild sensitivity to category growth.
    - last_month_sales uses a 30-day lag of brand_sales.
    - last_year_sales is a simulated reference value (because we only generate 365 days).
    """
    set_random_seed(config.seed)

    dates = pd.date_range(config.start_date, periods=config.n_days, freq="D")
    t = np.arange(config.n_days)

    seasonal = 1.0 + config.weekly_seasonality_strength * _weekly_seasonality(t)
    category_sales = (
        config.base_category_sales
        + config.daily_category_trend * t
        + config.base_category_sales * (seasonal - 1.0)
        + np.random.normal(0.0, config.category_noise_std, size=config.n_days)
    )
    category_sales = np.clip(category_sales, a_min=1_000.0, a_max=None)

    category_sales_series = pd.Series(category_sales, index=dates)
    category_growth = category_sales_series.pct_change().fillna(0.0)

    # Brand share: mostly stable, but can react to market momentum.
    share = (
        config.base_brand_share
        + config.share_sensitivity_to_growth * category_growth.values
        + np.random.normal(0.0, config.share_noise_std, size=config.n_days)
    )
    share = np.clip(share, config.min_share, config.max_share)

    brand_sales = category_sales * share + np.random.normal(0.0, 800.0, size=config.n_days)
    brand_sales = np.clip(brand_sales, a_min=100.0, a_max=None)

    df = pd.DataFrame(
        {
            "date": dates,
            "brand_sales": brand_sales,
            "category_sales": category_sales,
            "category_growth": category_growth.values,
        }
    )

    df["last_month_sales"] = df["brand_sales"].shift(30)

    # Simulated "last_year_sales": we build a plausible reference using current pattern + slight systematic drift.
    # This is intentionally not a perfect lag because we only have 1 year of data.
    drift = np.linspace(-0.03, 0.05, config.n_days)
    df["last_year_sales"] = (df["brand_sales"] * (1.0 - drift) + np.random.normal(0.0, 1200.0, config.n_days)).clip(
        lower=100.0
    )

    df["market_share"] = safe_divide(df["brand_sales"], df["category_sales"])
    return df


def save_raw_data(df: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    df.to_csv(output_path, index=False)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "data" / "raw_data.csv"

    df = generate_synthetic_data(GenerationConfig())
    save_raw_data(df, output_path)
    print(f"Saved synthetic dataset with {len(df)} rows to: {output_path}")


if __name__ == "__main__":
    main()
