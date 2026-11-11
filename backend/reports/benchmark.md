# Benchmark: first version vs rewrite

## 1. What the first version reported (80/20 split, one-step-ahead)

| dataset | best model (MAE) | seasonal naive on the same days (MAE) | model beats baseline? |
|---|---|---|---|
| synthetic (seed 0) | Gradient Boosting (83.9) | 80.7 | **no** |
| synthetic (seed 1) | Random Forest (82.7) | 87.4 | yes |
| synthetic (seed 2) | Gradient Boosting (79.7) | 84.1 | yes |

## 2. Honest evaluation: 30-day recursive forecasts, 5 rolling cut-offs

Skill = % lower MAE than the seasonal naive forecast (same day last week). MASE < 1 means better
than the in-sample weekly naive error.

### synthetic (seed 0): 730 days (2023-01-01 to 2024-12-30, 0 zero-sales days)

| forecaster | MAE | WAPE | MASE | skill vs seasonal naive |
|---|---|---|---|---|
| **Random Forest** | 61.5 | 4.5% | 0.89 | +61.8% |
| Gradient Boosting | 67.2 | 4.9% | 0.97 | +58.3% |
| Gradient Boosting (raw target) | 83.8 | 6.1% | 1.22 | +48.1% |
| Random Forest (raw target) | 109.8 | 8.0% | 1.59 | +31.9% |
| Linear Regression | 123.5 | 9.0% | 1.79 | +23.4% |
| Linear Regression (raw target) | 123.6 | 9.0% | 1.79 | +23.4% |
| Seasonal naive (same day last week) | 161.3 | 11.8% | 2.34 | +0.0% |
| Seasonal mean (last 4 same weekdays) | 170.9 | 12.5% | 2.48 | -6.0% |
| Naive (last value) | 202.2 | 14.8% | 2.93 | -25.3% |

### synthetic (seed 1): 730 days (2023-01-01 to 2024-12-30, 0 zero-sales days)

| forecaster | MAE | WAPE | MASE | skill vs seasonal naive |
|---|---|---|---|---|
| **Random Forest** | 66.9 | 4.9% | 1.04 | +60.7% |
| Gradient Boosting | 70.4 | 5.2% | 1.09 | +58.6% |
| Gradient Boosting (raw target) | 81.5 | 6.0% | 1.27 | +52.1% |
| Random Forest (raw target) | 102.4 | 7.5% | 1.59 | +39.8% |
| Linear Regression | 132.3 | 9.7% | 2.06 | +22.3% |
| Linear Regression (raw target) | 132.4 | 9.7% | 2.06 | +22.2% |
| Seasonal naive (same day last week) | 170.2 | 12.5% | 2.65 | +0.0% |
| Seasonal mean (last 4 same weekdays) | 173.6 | 12.7% | 2.70 | -2.0% |
| Naive (last value) | 188.6 | 13.8% | 2.93 | -10.8% |

### synthetic (seed 2): 730 days (2023-01-01 to 2024-12-30, 0 zero-sales days)

| forecaster | MAE | WAPE | MASE | skill vs seasonal naive |
|---|---|---|---|---|
| **Gradient Boosting** | 52.4 | 3.8% | 0.75 | +69.3% |
| Random Forest | 56.5 | 4.1% | 0.81 | +66.9% |
| Gradient Boosting (raw target) | 91.5 | 6.7% | 1.31 | +46.4% |
| Linear Regression | 137.8 | 10.0% | 1.97 | +19.3% |
| Linear Regression (raw target) | 137.9 | 10.1% | 1.97 | +19.3% |
| Random Forest (raw target) | 146.6 | 10.7% | 2.10 | +14.2% |
| Seasonal mean (last 4 same weekdays) | 169.2 | 12.3% | 2.42 | +1.0% |
| Seasonal naive (same day last week) | 170.8 | 12.4% | 2.44 | +0.0% |
| Naive (last value) | 189.7 | 13.8% | 2.71 | -11.1% |
