"""FastAPI service: upload daily sales, get a backtested 30-day forecast."""

from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.data import sample_sales
from app.forecasting import DataError, run, to_daily

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_DAYS = 3 * 366  # keeps a request within a few seconds on a small server
FAST = os.getenv("FAST_MODELS", "1") == "1"
FOLDS = int(os.getenv("BACKTEST_FOLDS", "3"))  # benchmarks use 5; 3 keeps the free server responsive

app = FastAPI(title="Retail Sales Forecasting API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class SalesPoint(BaseModel):
    date: str = Field(max_length=32)
    sales: float


def forecast_frame(df: pd.DataFrame) -> dict:
    try:
        series = to_daily(df)
    except DataError as exc:
        raise HTTPException(400, str(exc)) from exc
    if len(series.values) > MAX_DAYS:
        series.values = series.values.iloc[-MAX_DAYS:]
    return run(series, folds=FOLDS, fast=FAST)


@app.get("/")
def root():
    return {
        "message": "Retail Sales Forecasting API",
        "version": app.version,
        "endpoints": {
            "POST /upload": "CSV with date,sales columns",
            "POST /predict": "JSON list of {date, sales}",
            "GET /sample": "synthetic sample data",
            "GET /health": "health check",
        },
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


REAL_DATA = Path(__file__).resolve().parents[1] / "benchmarks" / "data" / "online_retail_ii_daily.csv"


@app.get("/sample")
def sample(days: int = 730, dataset: str = "synthetic"):
    """Synthetic data, or ``dataset=online-retail``: real daily sales of a UK online
    retailer, 2009-2011 (UCI Online Retail II, CC BY 4.0)."""
    if dataset == "online-retail":
        if not REAL_DATA.exists():
            raise HTTPException(404, "real dataset not available on this server")
        return {"data": pd.read_csv(REAL_DATA).to_dict("records"), "source": "UCI Online Retail II (CC BY 4.0)"}
    return {"data": sample_sales(days=max(120, min(days, MAX_DAYS)))}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "file is larger than 5 MB")
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(400, "could not read the CSV file") from exc
    df.columns = [str(c).strip().lower() for c in df.columns]
    return await run_in_threadpool(forecast_frame, df)


@app.post("/predict")
async def predict(points: list[SalesPoint]):
    if len(points) > 5000:
        raise HTTPException(413, "too many rows")
    df = pd.DataFrame([p.model_dump() for p in points])
    return await run_in_threadpool(forecast_frame, df)
