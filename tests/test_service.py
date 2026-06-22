"""API tests using FastAPI's TestClient against the trained bundle.

Skipped automatically if no artifact bundle has been trained yet.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bmmm import artifacts

_HAS_BUNDLE = (artifacts.DEFAULT_DIR / artifacts.MODEL_FILE).exists()
pytestmark = pytest.mark.skipif(not _HAS_BUNDLE, reason="no trained artifact bundle")


@pytest.fixture(scope="module")
def client() -> TestClient:
    from bmmm.service.app import app, get_state

    get_state.cache_clear()
    return TestClient(app)


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert len(body["channels"]) >= 1


def test_info(client: TestClient) -> None:
    r = client.get("/info")
    assert r.status_code == 200
    body = r.json()
    assert "diagnostics" in body and "fit_metrics" in body
    assert len(body["channel_stats"]) == len(body["channels"])


def test_predict(client: TestClient) -> None:
    channels = client.get("/health").json()["channels"]
    alloc = dict.fromkeys(channels, 300.0)
    r = client.post("/predict", json={"allocation": alloc})
    assert r.status_code == 200
    body = r.json()
    assert set(body["per_channel_response"]) == set(channels)
    assert body["total_response"] > 0


def test_predict_rejects_unknown_channel(client: TestClient) -> None:
    r = client.post("/predict", json={"allocation": {"nonexistent": 100.0}})
    assert r.status_code == 422


def test_optimize_budget(client: TestClient) -> None:
    r = client.post("/optimize-budget", json={"total_budget": 1500.0})
    assert r.status_code == 200
    body = r.json()
    assert abs(sum(body["allocation"].values()) - 1500.0) < 1.0
    assert "uplift_pct" in body
