# Market-Aware Business Forecasting System

An end-to-end, interview-ready analytics project to **forecast brand sales** and track **market share** using market context (category sales + category growth) and recent brand sales trends.

## Business Problem
You manage a brand inside a broader product category. You want to predict **future daily brand sales** while accounting for:
- **Category size** (total demand)
- **Category momentum** (growth/decline)
- **Recent brand performance** (last month, weekly effects)

This helps answer:
- Are we growing because the **market is growing**, or because the **brand is winning share**?
- What sales should we expect next month given current category conditions?

## What’s in This Project
- **Synthetic dataset** (365 daily rows) with realistic seasonality, trend, noise, and share dynamics
- **Modular pipeline**: preprocessing → feature engineering → linear regression → evaluation → plots → 30‑day forecast
- **Streamlit dashboard** for business stakeholders
- **Notebook** for analysis + storytelling

## Dataset
File: `data/raw_data.csv`

Columns:
- `date`
- `brand_sales`
- `category_sales`
- `category_growth` (day-over-day % change)
- `last_month_sales` (30-day lag of brand_sales)
- `last_year_sales` (simulated reference benchmark)
- `market_share` = `brand_sales / category_sales`

## Approach
### Preprocessing
Implemented in `src/data_preprocessing.py`:
- Parse `date` to datetime
- Sort by date (critical for time-series work)
- Handle missing values (forward fill appropriate lag fields)

### Feature Engineering
Implemented in `src/feature_engineering.py`:
- `month`
- `day_of_week`
- `lag_7`, `lag_30`
- `rolling_mean_7`

Rows with NA created by lags/rolling windows are dropped.

### Model
Implemented in `src/model.py` using **Linear Regression** (`scikit-learn`).

**Features used** (as requested):
- `category_sales`
- `category_growth`
- `last_month_sales`
- `last_year_sales`
- `lag_7`
- `lag_30`

**Target**:
- `brand_sales`

**Time-based split (no shuffle)**: the model trains on earlier dates and tests on the most recent window to mimic real forecasting.

### Evaluation
Implemented in `src/evaluate.py`:
- MAE
- RMSE

### Forecasting (Next 30 Days)
Implemented in `src/forecasting.py`:
- Forecast next 30 days using **last observed category signals** and recursively updated lag features (a practical baseline approach).

## Key Insights You Can Discuss in Interviews
- **Market-aware forecasting**: category sales and growth explain part of brand sales variation independent of brand execution.
- **Seasonality & recency**: weekly lag and last-month sales capture repeatable patterns (promotions, footfall, replenishment).
- **Market share as a KPI**: even if sales grow, share tells whether the brand is actually outperforming the category.

## How to Run
From the `business-forecasting/` directory:

### 1) Install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run the end-to-end pipeline (generates plots + prints metrics + prints 30‑day forecast)
```bash
python3 -m src.main
```

### 3) Launch the Streamlit dashboard
```bash
streamlit run app/app.py
```

### 4) Open the notebook
Open `notebooks/analysis.ipynb` in Jupyter / VS Code and run cells top-to-bottom.

## Project Structure
```
business-forecasting/
  data/
    raw_data.csv
  notebooks/
    analysis.ipynb
  src/
    data_generation.py
    data_preprocessing.py
    feature_engineering.py
    model.py
    evaluate.py
    forecasting.py
    visualization.py
    utils.py
    main.py
  app/
    app.py
  requirements.txt
  README.md
```

## Notes
- The included `data/raw_data.csv` makes the repo immediately runnable.
- Re-running `python3 -m src.main` will overwrite the CSV with a deterministic synthetic dataset (seeded), which is useful for reproducibility.

