import io

import pandas as pd
from fastapi.testclient import TestClient

from app.api import app
from app.data import sample_sales

client = TestClient(app)


def test_health_and_sample():
    assert client.get("/health").json() == {"status": "healthy"}
    data = client.get("/sample?days=200").json()["data"]
    assert len(data) == 200 and set(data[0]) == {"date", "sales"}


def test_predict_returns_forecast():
    response = client.post("/predict", json=sample_sales(days=200))
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["future_predictions"]) == 30 and body["best_model"]


def test_upload_csv_with_messy_headers():
    csv = pd.DataFrame(sample_sales(days=200)).rename(columns={"date": " Date", "sales": "SALES "}).to_csv(index=False)
    response = client.post("/upload", files={"file": ("s.csv", io.BytesIO(csv.encode()), "text/csv")})
    assert response.status_code == 200, response.text


def test_bad_input_is_a_400_not_a_500():
    short = client.post("/predict", json=sample_sales(days=40))
    assert short.status_code == 400 and "at least" in short.json()["detail"]
    wrong = client.post("/upload", files={"file": ("s.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")})
    assert wrong.status_code == 400 and "columns" in wrong.json()["detail"]


def test_real_dataset_endpoint():
    response = client.get("/sample?dataset=online-retail")
    assert response.status_code == 200
    body = response.json()
    assert len(body["data"]) > 500 and "UCI" in body["source"]
