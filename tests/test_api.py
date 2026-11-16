# tests/test_api.py
"""API contract tests with a fake model: no torch, no MLflow."""
import os

os.environ["SKIP_MODEL_LOAD"] = "1"

import pytest
from fastapi.testclient import TestClient

import serve


class FakePipe:
    def __call__(self, text, top_k=None):
        return [{"label": "joy", "score": 0.91}, {"label": "sadness", "score": 0.05},
                {"label": "anger", "score": 0.04}]


@pytest.fixture
def client():
    serve.STATE["pipe"] = FakePipe()
    serve.STATE["info"] = {"model": "tweet-emotion", "version": "7", "run_id": "abc"}
    yield TestClient(serve.app)
    serve.STATE["pipe"] = None


def test_health_reports_version(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["version"] == "7"


def test_predict_returns_top_label(client):
    r = client.post("/predict", json={"text": "i am so happy today"})
    body = r.json()
    assert r.status_code == 200
    assert body["label"] == "joy" and body["model_version"] == "7"
    assert set(body["scores"]) == {"joy", "sadness", "anger"}


def test_empty_text_rejected(client):
    assert client.post("/predict", json={"text": ""}).status_code == 422


def test_no_model_returns_503():
    serve.STATE["pipe"] = None
    assert TestClient(serve.app).get("/health").status_code == 503
