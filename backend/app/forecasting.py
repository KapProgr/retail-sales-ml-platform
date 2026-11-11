"""Daily sales forecasting with honest evaluation.

Design choices (each one fixes a problem found in the first version):

* **Baselines first.** Every result is compared with simple forecasts: the last
  value (naive), the same weekday last week (seasonal naive) and the mean of the
  last four same weekdays. The first version had none, and all three ML models were
  in fact worse than "same day last week".
* **Models learn the correction, not the level.** Tree models cannot predict values
  outside what they saw in training (e.g. a December peak or a growing trend), so
  the models predict ``sales - seasonal mean`` and the level comes from recent data.
* **Evaluation = how the forecast is used.** The app forecasts 30 days ahead, feeding
  its own predictions back as lags (recursive forecasting). The backtest does exactly
  that from several cut-off dates (rolling origin), instead of scoring one-step-ahead
  predictions that see the true previous day.
* **Scale-free metrics.** WAPE (MAPE breaks on zero-sales days) and MASE / skill
  relative to the seasonal naive forecast, so results on different datasets compare.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HORIZON = 30
MIN_HISTORY = 56  # days of history needed before the first forecasted day
LAGS = (1, 2, 7, 14, 21, 28)
FEATURES = [
    *(f"lag_{k}" for k in LAGS),
    "mean_7", "mean_28", "std_7", "same_weekday_mean_4",
    *(f"dow_{d}" for d in range(7)),
    "month_sin", "month_cos", "doy_sin", "doy_cos",
    "is_month_start", "is_month_end", "days_to_christmas",
]


# ----------------------------------------------------------------- data
class DataError(ValueError):
    """The uploaded data cannot be used (reported to the user as a 400)."""


@dataclass
class DailySeries:
    values: pd.Series  # float, daily DatetimeIndex without gaps
    filled_days: int  # dates missing from the input, filled with 0
    duplicate_dates: int  # dates that appeared more than once (summed)


def to_daily(df: pd.DataFrame, date_col: str = "date", value_col: str = "sales") -> DailySeries:
    """Validate raw rows and turn them into a gap-free daily series."""
    if date_col not in df.columns or value_col not in df.columns:
        raise DataError(f"the data needs '{date_col}' and '{value_col}' columns")
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "sales": pd.to_numeric(df[value_col], errors="coerce"),
        }
    ).dropna()
    if frame.empty:
        raise DataError("no valid rows: dates must be YYYY-MM-DD and sales numeric")
    frame["date"] = frame["date"].dt.normalize()
    counts = frame.groupby("date").size()
    daily = frame.groupby("date")["sales"].sum().sort_index()
    full = daily.asfreq("D")
    filled = int(full.isna().sum())
    full = full.fillna(0.0).astype(float)
    if len(full) < MIN_HISTORY + HORIZON:
        raise DataError(f"need at least {MIN_HISTORY + HORIZON} days of data, got {len(full)}")
    return DailySeries(full, filled, int((counts > 1).sum()))


# ----------------------------------------------------------------- features
def _calendar(dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    dow = dates.dayofweek.to_numpy()
    month = dates.month.to_numpy()
    doy = dates.dayofyear.to_numpy()
    christmas = pd.to_datetime([f"{y}-12-25" for y in dates.year])
    to_xmas = (christmas - dates).days.to_numpy()
    out = {f"dow_{d}": (dow == d).astype(float) for d in range(7)}
    out.update(
        month_sin=np.sin(2 * np.pi * month / 12),
        month_cos=np.cos(2 * np.pi * month / 12),
        doy_sin=np.sin(2 * np.pi * doy / 365.25),
        doy_cos=np.cos(2 * np.pi * doy / 365.25),
        is_month_start=dates.is_month_start.astype(float),
        is_month_end=dates.is_month_end.astype(float),
        days_to_christmas=np.where((to_xmas >= 0) & (to_xmas <= 60), to_xmas, 61).astype(float),
    )
    return out


def features_at(history: np.ndarray, dates: pd.DatetimeIndex, positions: np.ndarray) -> pd.DataFrame:
    """Features for the days at ``positions``, using only values *before* each day.

    ``history`` holds the series (actual values, extended with predictions during a
    recursive forecast); ``dates`` holds the matching dates.
    """
    rows: dict[str, list[float]] = {name: [] for name in FEATURES if not name.startswith(("dow_", "month_", "doy_", "is_", "days_"))}
    for p in positions:
        past = history[:p]
        for k in LAGS:
            rows[f"lag_{k}"].append(past[-k])
        rows["mean_7"].append(past[-7:].mean())
        rows["mean_28"].append(past[-28:].mean())
        rows["std_7"].append(past[-7:].std(ddof=1))
        rows["same_weekday_mean_4"].append(np.mean([past[-7], past[-14], past[-21], past[-28]]))
    frame = pd.DataFrame(rows)
    for name, values in _calendar(dates[positions]).items():
        frame[name] = values
    return frame[FEATURES]


# ----------------------------------------------------------------- forecasters
class Forecaster:
    name = "base"
    kind = "baseline"

    def fit(self, history: np.ndarray, dates: pd.DatetimeIndex) -> Forecaster:
        return self

    def forecast(self, history: np.ndarray, dates: pd.DatetimeIndex, horizon: int) -> np.ndarray:
        """``dates`` covers history + the horizon."""
        raise NotImplementedError


class Naive(Forecaster):
    name = "Naive (last value)"

    def forecast(self, history, dates, horizon):
        return np.repeat(history[-1], horizon)


class SeasonalNaive(Forecaster):
    name = "Seasonal naive (same day last week)"

    def forecast(self, history, dates, horizon):
        week = history[-7:]
        return np.array([week[h % 7] for h in range(horizon)])


class SeasonalMean(Forecaster):
    name = "Seasonal mean (last 4 same weekdays)"

    def forecast(self, history, dates, horizon):
        weeks = history[-28:].reshape(4, 7)
        profile = weeks.mean(axis=0)
        return np.array([profile[h % 7] for h in range(horizon)])


class MLForecaster(Forecaster):
    """Regression on engineered features, forecasting recursively.

    With ``residual=True`` (default) the model predicts the deviation from the
    seasonal mean; ``residual=False`` reproduces the first version (raw sales).
    """

    kind = "model"

    def __init__(self, name: str, estimator, residual: bool = True) -> None:
        self.name = name
        self.estimator = estimator
        self.residual = residual

    def fit(self, history, dates):
        positions = np.arange(MIN_HISTORY, len(history))
        x = features_at(history, dates, positions)
        y = history[positions] - (x["same_weekday_mean_4"].to_numpy() if self.residual else 0.0)
        self.estimator.fit(x, y)
        return self

    def forecast(self, history, dates, horizon):
        extended = np.concatenate([history, np.zeros(horizon)])
        start = len(history)
        for h in range(horizon):
            x = features_at(extended, dates, np.array([start + h]))
            pred = float(self.estimator.predict(x)[0])
            if self.residual:
                pred += float(x["same_weekday_mean_4"].iloc[0])
            extended[start + h] = max(pred, 0.0)  # sales cannot be negative
        return extended[start:]

    def importance(self) -> list[dict]:
        model = self.estimator[-1] if hasattr(self.estimator, "steps") else self.estimator
        if hasattr(model, "feature_importances_"):
            weights = np.asarray(model.feature_importances_)
        elif hasattr(model, "coef_"):
            weights = np.abs(np.asarray(model.coef_))
        else:
            return []
        top = weights.max() or 1.0
        ranked = sorted(zip(FEATURES, weights, strict=True), key=lambda t: t[1], reverse=True)[:10]
        return [{"feature": f, "importance": float(w / top * 100)} for f, w in ranked]


def default_forecasters(residual: bool = True, fast: bool = False) -> list[Forecaster]:
    trees = 60 if fast else 150
    suffix = "" if residual else " (raw target)"
    return [
        Naive(),
        SeasonalNaive(),
        SeasonalMean(),
        MLForecaster("Linear Regression" + suffix, make_pipeline(StandardScaler(), Ridge(alpha=1.0)), residual),
        MLForecaster(
            "Random Forest" + suffix,
            RandomForestRegressor(n_estimators=trees, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=1),
            residual,
        ),
        MLForecaster(
            "Gradient Boosting" + suffix,
            GradientBoostingRegressor(n_estimators=trees, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42),
            residual,
        ),
    ]


# ----------------------------------------------------------------- evaluation
@dataclass
class FoldResult:
    origin: pd.Timestamp
    actual: np.ndarray
    predictions: dict[str, np.ndarray]


@dataclass
class Backtest:
    horizon: int
    folds: list[FoldResult]
    scale: float  # in-sample MAE of the seasonal naive forecast (for MASE)
    metrics: dict[str, dict] = field(default_factory=dict)


def metrics(actual: np.ndarray, predicted: np.ndarray, scale: float, baseline_mae: float | None = None) -> dict:
    err = actual - predicted
    mae = float(np.mean(np.abs(err)))
    total = float(np.sum(np.abs(actual)))
    nonzero = actual != 0
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "mae": mae,
        "rmse": float(np.sqrt(np.mean(err**2))),
        "wape": float(np.sum(np.abs(err)) / total * 100) if total else float("nan"),
        "mape": float(np.mean(np.abs(err[nonzero] / actual[nonzero])) * 100) if nonzero.any() else float("nan"),
        "r2": float(1 - np.sum(err**2) / ss_tot) if ss_tot else float("nan"),
        "mase": mae / scale if scale else float("nan"),
        "skill_vs_seasonal_naive": (1 - mae / baseline_mae) * 100 if baseline_mae else 0.0,
    }


def fold_origins(n: int, horizon: int, folds: int) -> list[int]:
    """Cut-off positions for the last ``folds`` non-overlapping horizons."""
    origins = [n - horizon * k for k in range(folds, 0, -1)]
    origins = [o for o in origins if o >= MIN_HISTORY + 28]
    if not origins:
        raise DataError("not enough history for a backtest")
    return origins


def backtest(series: pd.Series, forecasters: list[Forecaster], horizon: int = HORIZON, folds: int = 5) -> Backtest:
    values, dates = series.to_numpy(dtype=float), series.index
    origins = fold_origins(len(values), horizon, folds)
    first = origins[0]
    scale = float(np.mean(np.abs(values[7:first] - values[: first - 7])))
    results = []
    for origin in origins:
        history = values[:origin]
        window = dates[: origin + horizon]
        preds = {}
        for f in forecasters:
            f.fit(history, window[:origin])
            preds[f.name] = f.forecast(history, window, horizon)
        results.append(FoldResult(dates[origin], values[origin : origin + horizon], preds))
    bt = Backtest(horizon, results, scale)
    actual = np.concatenate([r.actual for r in results])
    snaive = SeasonalNaive.name
    base_mae = None
    if any(snaive in r.predictions for r in results):
        base_mae = float(np.mean(np.abs(actual - np.concatenate([r.predictions[snaive] for r in results]))))
    for f in forecasters:
        pred = np.concatenate([r.predictions[f.name] for r in results])
        bt.metrics[f.name] = {"kind": f.kind, **metrics(actual, pred, scale, base_mae)}
    return bt


# ----------------------------------------------------------------- end to end
def run(series: DailySeries, horizon: int = HORIZON, folds: int = 5, fast: bool = False) -> dict:
    """Backtest all forecasters, pick the best by MAE, refit it on all data and
    forecast ``horizon`` days with an empirical 80% interval."""
    s = series.values
    forecasters = default_forecasters(fast=fast)
    bt = backtest(s, forecasters, horizon, folds)
    best_name = min(bt.metrics, key=lambda name: bt.metrics[name]["mae"])
    best = next(f for f in forecasters if f.name == best_name)

    values = s.to_numpy(dtype=float)
    future_dates = pd.date_range(s.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    all_dates = s.index.append(future_dates)
    best.fit(values, s.index)
    future = best.forecast(values, all_dates, horizon)

    residuals = np.concatenate([r.actual - r.predictions[best_name] for r in bt.folds])
    low, high = np.quantile(residuals, [0.1, 0.9])

    last = bt.folds[-1]
    test_dates = pd.date_range(last.origin, periods=horizon, freq="D")
    ml = [f for f in forecasters if isinstance(f, MLForecaster)]
    explain = best if isinstance(best, MLForecaster) else min(ml, key=lambda f: bt.metrics[f.name]["mae"])
    if explain is not best:
        explain.fit(values, s.index)

    return {
        "models": [{"name": name, **m} for name, m in bt.metrics.items()],
        "best_model": best_name,
        "test_predictions": [
            {
                "date": d.strftime("%Y-%m-%d"),
                "actual": float(a),
                "predicted": float(p),
                "baseline": float(b),
            }
            for d, a, p, b in zip(test_dates, last.actual, last.predictions[best_name], last.predictions[SeasonalNaive.name], strict=True)
        ],
        "future_predictions": [
            {
                "date": d.strftime("%Y-%m-%d"),
                "predicted": float(p),
                "lower": float(max(p + low, 0.0)),
                "upper": float(p + high),
            }
            for d, p in zip(future_dates, future, strict=True)
        ],
        "feature_importance": explain.importance(),
        "feature_importance_model": explain.name,
        "backtest": {
            "horizon_days": horizon,
            "folds": len(bt.folds),
            "cutoffs": [f.origin.strftime("%Y-%m-%d") for f in bt.folds],
        },
        "data": {
            "days": int(len(s)),
            "start": s.index[0].strftime("%Y-%m-%d"),
            "end": s.index[-1].strftime("%Y-%m-%d"),
            "filled_days": series.filled_days,
            "duplicate_dates": series.duplicate_dates,
        },
    }
