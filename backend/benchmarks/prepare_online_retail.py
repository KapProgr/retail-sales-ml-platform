"""Build benchmarks/data/online_retail_ii_daily.csv from the UCI download.

Source: "Online Retail II", UCI Machine Learning Repository (Daqing Chen, 2012),
https://archive.ics.uci.edu/dataset/502/online+retail+ii, licence CC BY 4.0.
Transactions of a UK online retailer, Dec 2009 - Dec 2011.

    python -m benchmarks.prepare_online_retail path/to/online+retail+ii.zip

Daily sales = sum of Quantity * Price over sale lines. Cancellations (invoice
starting with "C"), bad-debt adjustments ("A") and non-positive quantities or prices
are excluded. Days without transactions (the shop does not trade on Saturdays and
closes over Christmas) become 0 in the daily series.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

OUT = Path(__file__).with_name("data") / "online_retail_ii_daily.csv"


def load(path: Path) -> pd.DataFrame:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith(".xlsx"))
            raw = z.read(name)
        sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None, engine="openpyxl")
    else:
        sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    return pd.concat(sheets.values(), ignore_index=True)


def daily(df: pd.DataFrame) -> pd.DataFrame:
    invoice = df["Invoice"].astype(str)
    sales = df[~invoice.str.startswith(("C", "A")) & (df["Quantity"] > 0) & (df["Price"] > 0)].copy()
    sales["date"] = pd.to_datetime(sales["InvoiceDate"]).dt.normalize()
    sales["revenue"] = sales["Quantity"] * sales["Price"]
    out = sales.groupby("date")["revenue"].sum().round(2).rename("sales").reset_index()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("online+retail+ii.zip")
    df = load(path)
    out = daily(df)
    OUT.parent.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"{len(df):,} transaction lines -> {len(out)} trading days ({out['date'].min()} to {out['date'].max()}): {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
