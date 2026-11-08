"""Endpoint tests for the FastAPI prediction service."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import main as api
from src.config import MODELS_DIR, PROCESSED_DATA_DIR, TEST_FILE


@pytest.mark.integration
def test_health_with_real_trained_model():
    """Check the health endpoint after the real model loads at startup."""
    model_path = MODELS_DIR / "random_forest.pkl"
    if not model_path.exists():
        pytest.skip("Real model is unavailable; run dvc pull or dvc repro")

    with TestClient(api.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": "true"}


@pytest.mark.integration
def test_predict_with_real_trained_model():
    """Exercise the HTTP endpoint with the trained artifact and a held-out row."""
    model_path = MODELS_DIR / "random_forest.pkl"
    test_path = PROCESSED_DATA_DIR / TEST_FILE
    if not model_path.exists() or not test_path.exists():
        pytest.skip("Real model or processed test data is unavailable; run dvc pull or dvc repro")

    row = pd.read_csv(test_path).iloc[0]
    genre_columns = [column for column in row.index if column.startswith("genre_")]
    payload = {
        "danceability": float(row["danceability"]),
        "energy": float(row["energy"]),
        "key": int(row["key"]),
        "loudness": float(row["loudness"]),
        "mode": int(row["mode"]),
        "speechiness": float(row["speechiness"]),
        "acousticness": float(row["acousticness"]),
        "instrumentalness": float(row["instrumentalness"]),
        "liveness": float(row["liveness"]),
        "valence": float(row["valence"]),
        "tempo": float(row["tempo"]),
        "duration_ms": int(row["duration_ms"]),
        "explicit": int(row["explicit"]),
        "genre": next(
            (column.removeprefix("genre_") for column in genre_columns if row[column] == 1),
            "unknown",
        ),
    }

    with TestClient(api.app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200, response.text
    prediction = response.json()["predicted_popularity"]
    assert np.isfinite(prediction)
    assert 0 <= prediction <= 100
