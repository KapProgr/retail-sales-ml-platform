"""Synthetic sample data (same shape as the first version, now reproducible)."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np


def sample_sales(days: int = 730, seed: int = 42, start: str = "2023-01-01") -> list[dict]:
    """Daily sales: yearly cycle, weekend uplift, December peak, trend and noise."""
    rng = np.random.default_rng(seed)
    first = datetime.fromisoformat(start)
    rows = []
    for i in range(days):
        date = first + timedelta(days=i)
        sales = 1000.0
        sales += np.sin(i / 365 * 2 * np.pi) * 200
        sales += 300 if date.weekday() in (5, 6) else 0
        sales += 500 if date.month == 12 else 0
        sales += rng.normal(0, 50)
        sales += i * 0.5
        rows.append({"date": date.strftime("%Y-%m-%d"), "sales": round(float(sales), 2)})
    return rows
