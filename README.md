# MixWise — Marketing Mix Modeling

A web app for Marketing Mix Modeling (MMM) built with Streamlit and Python.

## Features

- **Data Maturity Tier Detection** — automatically classifies your data as Full MMM (Tier 1), Lite MMM (Tier 2), or Incrementality (Tier 3)
- **Model Builder** — OLS and Ridge regression models with channel contribution analysis
- **Priors & Adstock** — configure adstock decay and Hill saturation per channel
- **ROAS Dashboard** — revenue attribution, budget optimizer, and waterfall chart
- **A/B Testing** — coming soon

## Local Setup

### Requirements
- Python 3.11+

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run artifacts/mixwise/app.py
```

The app will open at `http://localhost:8501` in your browser.

## Data Format

Upload a CSV with weekly marketing data. Expected columns:

| Column | Description |
|--------|-------------|
| `date` | Week start date (YYYY-MM-DD) |
| `tv_spend` | TV advertising spend ($) |
| `paid_search_spend` | Paid search spend ($) |
| `social_spend` | Social media spend ($) |
| `display_spend` | Display advertising spend ($) |
| `revenue` | Weekly revenue ($) |
| `promo_flag` | 1 if promotional week, 0 otherwise (optional) |
| `seasonality_index` | Seasonal index 0–1 (optional) |

If no file is uploaded, the app loads a built-in 104-week synthetic demo dataset.

## Tier Detection

| Tier | Requirement | Mode |
|------|-------------|------|
| 1 — Full MMM | 104+ weeks & 3+ channels | Full regression model |
| 2 — Lite MMM | 26+ weeks or 2+ channels | Regression with industry priors |
| 3 — Incrementality | Less than above | Pre/post lift calculator |
