# Benchmark: first version vs rewrite

## 1. What the first version reported (80/20 split, one-step-ahead)

| dataset | best model (MAE) | seasonal naive on the same days (MAE) | model beats baseline? |
|---|---|---|---|
| synthetic 365d (seed 0) | Gradient Boosting (204.6) | 104.2 | **no** |
| synthetic 365d (seed 1) | Random Forest (204.3) | 100.8 | **no** |
| synthetic 365d (seed 2) | Random Forest (198.5) | 112.0 | **no** |
| synthetic 730d (seed 0) | Gradient Boosting (83.9) | 80.7 | **no** |
| synthetic 730d (seed 1) | Random Forest (82.7) | 87.4 | yes |
| synthetic 730d (seed 2) | Gradient Boosting (79.7) | 84.1 | yes |
| Online Retail II (UCI, real) | Linear Regression (12,938.6) | 15,304.0 | yes |

## 2. Honest evaluation: 30-day recursive forecasts, 5 rolling cut-offs

Skill = % lower MAE than the seasonal naive forecast (same day last week). MASE < 1 means better
than the in-sample weekly naive error.

### synthetic 365d (seed 0): 365 days (2023-01-01 to 2023-12-31, 0 zero-sales days)

| forecaster | MAE | WAPE | MASE | skill vs seasonal naive |
|---|---|---|---|---|
| **Gradient Boosting** | 95.7 | 8.1% | 1.60 | +44.6% |
| Random Forest | 126.8 | 10.7% | 2.12 | +26.6% |
| Gradient Boosting (raw target) | 129.7 | 10.9% | 2.17 | +24.9% |
| Linear Regression | 131.5 | 11.1% | 2.20 | +23.8% |
| Linear Regression (raw target) | 131.5 | 11.1% | 2.20 | +23.8% |
| Random Forest (raw target) | 148.5 | 12.5% | 2.48 | +14.0% |
| Seasonal naive (same day last week) | 172.6 | 14.5% | 2.89 | +0.0% |
| Naive (last value) | 172.9 | 14.6% | 2.89 | -0.2% |
| Seasonal mean (last 4 same weekdays) | 179.4 | 15.1% | 3.00 | -3.9% |

### synthetic 365d (seed 1): 365 days (2023-01-01 to 2023-12-31, 0 zero-sales days)

| forecaster | MAE | WAPE | MASE | skill vs seasonal naive |
|---|---|---|---|---|
| **Gradient Boosting** | 73.4 | 6.2% | 1.33 | +51.2% |
| Random Forest | 123.0 | 10.4% | 2.23 | +18.2% |
| Gradient Boosting (raw target) | 123.2 | 10.4% | 2.23 | +18.1% |
| Linear Regression (raw target) | 127.6 | 10.8% | 2.31 | +15.2% |
| Linear Regression | 127.8 | 10.8% | 2.32 | +15.0% |
| Random Forest (raw target) | 145.4 | 12.3% | 2.64 | +3.3% |
| Seasonal naive (same day last week) | 150.4 | 12.7% | 2.73 | +0.0% |
| Naive (last value) | 172.7 | 14.6% | 3.13 | -14.8% |
| Seasonal mean (last 4 same weekdays) | 172.8 | 14.6% | 3.13 | -14.9% |

### synthetic 365d (seed 2): 365 days (2023-01-01 to 2023-12-31, 0 zero-sales days)

| forecaster | MAE | WAPE | MASE | skill vs seasonal naive |
|---|---|---|---|---|
| **Gradient Boosting** | 93.7 | 7.9% | 1.60 | +40.3% |
| Gradient Boosting (raw target) | 127.3 | 10.7% | 2.17 | +18.9% |
| Random Forest | 145.5 | 12.2% | 2.49 | +7.3% |
| Random Forest (raw target) | 156.6 | 13.2% | 2.67 | +0.3% |
| Seasonal naive (same day last week) | 157.0 | 13.2% | 2.68 | +0.0% |
| Naive (last value) | 162.7 | 13.7% | 2.78 | -3.7% |
| Linear Regression | 174.2 | 14.6% | 2.98 | -11.0% |
| Linear Regression (raw target) | 174.4 | 14.7% | 2.98 | -11.1% |
| Seasonal mean (last 4 same weekdays) | 181.1 | 15.2% | 3.09 | -15.3% |

### synthetic 730d (seed 0): 730 days (2023-01-01 to 2024-12-30, 0 zero-sales days)

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

### synthetic 730d (seed 1): 730 days (2023-01-01 to 2024-12-30, 0 zero-sales days)

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

### synthetic 730d (seed 2): 730 days (2023-01-01 to 2024-12-30, 0 zero-sales days)

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

### Online Retail II (UCI, real): 739 days (2009-12-01 to 2011-12-09, 135 zero-sales days)

| forecaster | MAE | WAPE | MASE | skill vs seasonal naive |
|---|---|---|---|---|
| **Gradient Boosting** | 11,089.0 | 29.8% | 0.92 | +13.7% |
| Random Forest (raw target) | 11,134.3 | 29.9% | 0.92 | +13.4% |
| Random Forest | 11,488.8 | 30.9% | 0.95 | +10.6% |
| Seasonal mean (last 4 same weekdays) | 12,572.2 | 33.8% | 1.04 | +2.2% |
| Seasonal naive (same day last week) | 12,851.7 | 34.5% | 1.06 | +0.0% |
| Gradient Boosting (raw target) | 12,885.5 | 34.6% | 1.07 | -0.3% |
| Linear Regression (raw target) | 18,325.3 | 49.2% | 1.52 | -42.6% |
| Linear Regression | 18,327.1 | 49.2% | 1.52 | -42.6% |
| Naive (last value) | 29,105.4 | 78.2% | 2.41 | -126.5% |
