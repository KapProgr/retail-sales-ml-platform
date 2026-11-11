# Retail Sales Forecasting

Upload daily sales, get a 30-day forecast with an uncertainty band, and see how
every model compares with simple baselines on the same backtest.

FastAPI + scikit-learn backend, React + Recharts frontend, both in this repository.

## Results

<!-- RESULTS:START -->
Run `python -m benchmarks.run_benchmark` in `backend/` to fill this section.
<!-- RESULTS:END -->

## What changed from the first version, and why

The first version trained three models on one 80/20 split and reported their error
on the test days. Measuring it properly showed three problems:

| Problem | Effect | Fix |
|---|---|---|
| No baseline | The models looked fine, but on its own sample data they lost to "same day last week" (see Results) | Three baselines are always evaluated and can win; the app picks whatever is best |
| Wrong test | Each test day was scored with the *true* previous days as inputs, while the real 30-day forecast feeds back its own predictions | Rolling-origin backtest: 30-day recursive forecasts from several cut-off dates, exactly like production |
| Trees cannot extrapolate | Random forest / boosting cannot predict above the highest value they saw (December peak, growth) | Models predict the deviation from the recent same-weekday average; the level comes from the data |

Also: WAPE instead of MAPE (zero-sales days), MASE and skill vs the seasonal naive
forecast, an 80% interval from backtest errors, input validation with clear 400
errors, and the frontend no longer shows **simulated** results when the API is
unavailable (it used to invent metrics and predictions with `Math.random`).

## How it works

```mermaid
flowchart LR
    A[CSV / JSON<br/>date, sales] --> B[to_daily<br/>validate, sum duplicates,<br/>fill missing days]
    B --> C[Backtest<br/>cut-offs t1..tk]
    C --> D1[Baselines<br/>naive, seasonal naive,<br/>seasonal mean]
    C --> D2[Ridge / Random Forest /<br/>Gradient Boosting<br/>target = sales - seasonal mean]
    D1 & D2 --> E[MAE, WAPE, MASE,<br/>skill vs baseline]
    E --> F[Best by MAE<br/>refit on all data]
    F --> G[30-day recursive forecast<br/>+ 80% interval]
```

Features use only information available before the forecasted day (lags 1, 2, 7, 14,
21, 28; 7/28-day means; weekday; month and day-of-year cycles; days to Christmas).
A test checks this: changing the target day and everything after it does not change
its features.

## Run locally

```bash
# backend
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows (source .venv/bin/activate elsewhere)
pip install -r requirements-dev.txt
uvicorn main:app --reload        # http://localhost:8000/docs
pytest -q

# frontend
cd frontend
npm install
copy .env.example .env           # VITE_API_URL=http://localhost:8000
npm run dev                      # http://localhost:5173
```

### Benchmark

```bash
cd backend
python -m benchmarks.prepare_online_retail path\to\online+retail+ii.zip   # once
python -m benchmarks.run_benchmark
```

Writes `reports/benchmark.md` and `reports/benchmark.json`. The original pipeline is
kept in `benchmarks/original_pipeline.py` to reproduce the "before" numbers.

## Data format

```csv
date,sales
2023-01-01,1200.50
2023-01-02,1350.75
```

At least 86 days. Duplicate dates are summed, missing dates count as zero sales,
header names are case-insensitive.

## Deployment

- **API (Render):** `render.yaml` (root directory `backend`). The free plan sleeps
  when idle; the first request takes about a minute.
- **Frontend (Vercel):** root directory `frontend`, framework Vite, environment
  variable `VITE_API_URL`.

## Repository layout

```
backend/
  app/forecasting.py     data validation, features, baselines, models, backtest
  app/api.py             FastAPI endpoints
  benchmarks/            before/after benchmark, original pipeline, real dataset prep
  tests/
frontend/src/App.jsx     upload, model comparison, forecast charts
```

The backend and frontend previously lived in `RetailSalesML/retail-sales-api` and
`RetailSalesML/retail-sales-frontend`; their history is merged here.

## Data credits

Online Retail II: Chen, D. (2012), UCI Machine Learning Repository,
https://doi.org/10.24432/C5CG6D, CC BY 4.0.

## License

MIT
