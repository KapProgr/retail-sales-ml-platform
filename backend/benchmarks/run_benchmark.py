"""Before/after benchmark.

    python -m benchmarks.run_benchmark            # all datasets in benchmarks/data
    python -m benchmarks.prepare_online_retail    # once, to build the real dataset

For every dataset it reports
  1. what the first version of the app showed: its own 80/20 split, scored
     one step ahead (each test day sees the true previous days),
  2. the seasonal-naive forecast on that same split, and
  3. the honest evaluation: 30-day recursive forecasts from 5 rolling cut-offs, for
     the baselines, the original raw-target models and the new residual models.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from app.data import sample_sales  # noqa: E402
from app.forecasting import backtest, default_forecasters, to_daily  # noqa: E402
from benchmarks import original_pipeline  # noqa: E402

DATA = Path(__file__).with_name("data")
REPORTS = ROOT / "reports"


def datasets() -> dict[str, pd.DataFrame]:
    # 365 days = the first version's sample data; 730 days = the new default (two Decembers)
    sets = {f"synthetic {d}d (seed {s})": pd.DataFrame(sample_sales(days=d, seed=s)) for d in (365, 730) for s in (0, 1, 2)}
    real = DATA / "online_retail_ii_daily.csv"
    if real.exists():
        sets["Online Retail II (UCI, real)"] = pd.read_csv(real)
    return sets


def original_view(df: pd.DataFrame) -> dict:
    """The first version's own numbers, and the seasonal naive on its test days."""
    predictor = original_pipeline.MLPredictor()
    out = predictor.train_and_predict(df[["date", "sales"]].copy())
    test = pd.DataFrame(out["test_predictions"])
    series = to_daily(df).values
    dates = pd.to_datetime(test["date"])
    snaive = series.shift(7).loc[dates].to_numpy()
    actual = test["actual"].to_numpy()
    best = min(out["models"], key=lambda m: m["mae"])
    return {
        "models": {m["name"]: round(m["mae"], 1) for m in out["models"]},
        "best_model": best["name"],
        "best_mae": round(best["mae"], 1),
        "seasonal_naive_mae": round(float(np.mean(np.abs(actual - snaive))), 1),
        "test_days": len(test),
    }


def evaluate(name: str, df: pd.DataFrame) -> dict:
    started = time.time()
    series = to_daily(df)
    old = original_view(df)
    forecasters = default_forecasters(residual=True)
    forecasters += [f for f in default_forecasters(residual=False) if f.kind == "model"]
    bt = backtest(series.values, forecasters)
    rows = {n: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()} for n, m in bt.metrics.items()}
    print(f"{name}: {time.time() - started:.0f}s")
    return {
        "dataset": name,
        "days": len(series.values),
        "start": str(series.values.index[0].date()),
        "end": str(series.values.index[-1].date()),
        "zero_days": int((series.values == 0).sum()),
        "original_app": old,
        "backtest": rows,
        "cutoffs": [str(f.origin.date()) for f in bt.folds],
    }


def to_markdown(results: list[dict]) -> str:
    lines = ["# Benchmark: first version vs rewrite", ""]
    lines += [
        "## 1. What the first version reported (80/20 split, one-step-ahead)",
        "",
        "| dataset | best model (MAE) | seasonal naive on the same days (MAE) | model beats baseline? |",
        "|---|---|---|---|",
    ]
    for r in results:
        o = r["original_app"]
        beats = "yes" if o["best_mae"] < o["seasonal_naive_mae"] else "**no**"
        lines.append(f"| {r['dataset']} | {o['best_model']} ({o['best_mae']:,.1f}) | {o['seasonal_naive_mae']:,.1f} | {beats} |")
    lines += [
        "",
        "## 2. Honest evaluation: 30-day recursive forecasts, 5 rolling cut-offs",
        "",
        "Skill = % lower MAE than the seasonal naive forecast (same day last week). MASE < 1 means better",
        "than the in-sample weekly naive error.",
    ]
    for r in results:
        lines += [
            "",
            f"### {r['dataset']}: {r['days']} days ({r['start']} to {r['end']}, {r['zero_days']} zero-sales days)",
            "",
            "| forecaster | MAE | WAPE | MASE | skill vs seasonal naive |",
            "|---|---|---|---|---|",
        ]
        ranked = sorted(r["backtest"].items(), key=lambda kv: kv[1]["mae"])
        for name, m in ranked:
            label = f"**{name}**" if name == ranked[0][0] else name
            lines.append(f"| {label} | {m['mae']:,.1f} | {m['wape']:.1f}% | {m['mase']:.2f} | {m['skill_vs_seasonal_naive']:+.1f}% |")
    return "\n".join(lines) + "\n"


def summary(results: list[dict]) -> dict:
    """Average skill per forecaster across datasets (for the README)."""
    names = results[0]["backtest"].keys()
    return {n: round(float(np.mean([r["backtest"][n]["skill_vs_seasonal_naive"] for r in results])), 1) for n in names}


def main() -> int:
    results = [evaluate(name, df) for name, df in datasets().items()]
    REPORTS.mkdir(exist_ok=True)
    payload = {"results": results, "mean_skill": summary(results)}
    (REPORTS / "benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (REPORTS / "benchmark.md").write_text(to_markdown(results), encoding="utf-8")
    print((REPORTS / "benchmark.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
