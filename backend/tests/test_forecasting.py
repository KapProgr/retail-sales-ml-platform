import numpy as np
import pandas as pd
import pytest

from app.data import sample_sales
from app.forecasting import (
    FEATURES,
    DataError,
    MLForecaster,
    Naive,
    SeasonalMean,
    SeasonalNaive,
    backtest,
    default_forecasters,
    features_at,
    metrics,
    run,
    to_daily,
)


def weekly_series(weeks=20, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    pattern = np.array([10, 20, 30, 40, 50, 60, 70], dtype=float)
    values = np.tile(pattern, weeks) + rng.normal(0, noise, 7 * weeks)
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=7 * weeks, freq="D"))


def test_to_daily_fills_gaps_and_sums_duplicates():
    rows = [{"date": str(d.date()), "sales": 1.0} for d in pd.date_range("2024-01-01", periods=100)]
    rows.append({"date": "2024-01-01", "sales": 4.0})  # duplicate date
    rows = [r for r in rows if r["date"] != "2024-02-01"]  # missing day
    daily = to_daily(pd.DataFrame(rows))
    assert daily.values.loc["2024-01-01"] == 5.0
    assert daily.values.loc["2024-02-01"] == 0.0
    assert daily.filled_days == 1 and daily.duplicate_dates == 1


@pytest.mark.parametrize(
    "frame,message",
    [
        (pd.DataFrame({"day": ["2024-01-01"], "sales": [1]}), "columns"),
        (pd.DataFrame({"date": ["x", "y"], "sales": ["a", "b"]}), "no valid rows"),
        (pd.DataFrame({"date": pd.date_range("2024-01-01", periods=30).astype(str), "sales": 1.0}), "at least"),
    ],
)
def test_bad_data_is_rejected_clearly(frame, message):
    with pytest.raises(DataError, match=message):
        to_daily(frame)


def test_seasonal_baselines_repeat_the_weekly_pattern():
    s = weekly_series()
    values, dates = s.to_numpy(), s.index
    future = dates.append(pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=14))
    expected = np.tile([10, 20, 30, 40, 50, 60, 70], 2)
    assert np.allclose(SeasonalNaive().forecast(values, future, 14), expected)
    assert np.allclose(SeasonalMean().forecast(values, future, 14), expected)
    assert np.allclose(Naive().forecast(values, future, 3), [70, 70, 70])


def test_features_use_only_the_past():
    s = weekly_series()
    values = s.to_numpy().copy()
    x1 = features_at(values, s.index, np.array([100]))
    values[100:] = 1e9  # changing the target day and the future must not change its features
    x2 = features_at(values, s.index, np.array([100]))
    pd.testing.assert_frame_equal(x1, x2)
    assert list(x1.columns) == FEATURES


def test_models_learn_a_perfect_weekly_pattern():
    s = weekly_series(weeks=30)
    bt = backtest(s, [SeasonalNaive(), *[f for f in default_forecasters(fast=True) if f.kind == "model"]], folds=2)
    for m in bt.metrics.values():
        assert m["mae"] < 1.0


def test_residual_target_follows_growth_that_raw_trees_cannot():
    """Trees cannot predict above the highest value they saw; with a growing trend the
    raw-target forest flattens out, the residual one keeps up."""
    rng = np.random.default_rng(1)
    days = pd.date_range("2024-01-01", periods=300, freq="D")
    s = pd.Series(100 + 2.0 * np.arange(300) + rng.normal(0, 2, 300), index=days)
    from sklearn.ensemble import RandomForestRegressor

    raw = MLForecaster("raw", RandomForestRegressor(n_estimators=30, random_state=0), residual=False)
    res = MLForecaster("res", RandomForestRegressor(n_estimators=30, random_state=0), residual=True)
    bt = backtest(s, [raw, res], folds=1)  # forecast days 270-299, all above the training maximum
    assert bt.metrics["res"]["mae"] < 0.5 * bt.metrics["raw"]["mae"]


def test_metrics_handle_zero_sales_days():
    m = metrics(np.array([0.0, 10.0, 10.0]), np.array([1.0, 9.0, 11.0]), scale=2.0, baseline_mae=2.0)
    assert m["mae"] == pytest.approx(1.0)
    assert m["wape"] == pytest.approx(15.0)
    assert m["mape"] == pytest.approx(10.0)  # zero day excluded
    assert m["mase"] == pytest.approx(0.5)
    assert m["skill_vs_seasonal_naive"] == pytest.approx(50.0)


def test_run_returns_a_complete_report():
    out = run(to_daily(pd.DataFrame(sample_sales(days=400))), folds=2, fast=True)
    names = {m["name"] for m in out["models"]}
    assert SeasonalNaive.name in names and "Gradient Boosting" in names
    assert out["best_model"] in names
    assert len(out["future_predictions"]) == 30
    assert all(p["lower"] <= p["predicted"] <= p["upper"] for p in out["future_predictions"])
    assert len(out["test_predictions"]) == 30 and "baseline" in out["test_predictions"][0]
    assert out["feature_importance"]
    assert out["backtest"]["folds"] == 2


def test_sample_data_is_reproducible():
    assert sample_sales(days=10, seed=3) == sample_sales(days=10, seed=3)
    assert sample_sales(days=10, seed=3) != sample_sales(days=10, seed=4)
