# tests/test_api_onnx.py
"""ONNX API contract tests with a fake classifier: no onnxruntime, no MLflow."""
import os

os.environ["SKIP_MODEL_LOAD"] = "1"

import pytest
from fastapi.testclient import TestClient

import serve_onnx as serve


class FakeClassifier:
    def scores(self, text):
        return {"joy": 0.91, "sadness": 0.05, "anger": 0.04}


@pytest.fixture
def client():
    serve.STATE["clf"] = FakeClassifier()
    serve.STATE["info"] = {"model": "tweet-emotion-onnx", "version": "3", "run_id": "abc"}
    yield TestClient(serve.app)
    serve.STATE["clf"] = None


def test_health_reports_version(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["version"] == "3"


def test_predict_returns_top_label(client):
    r = client.post("/predict", json={"text": "i am so happy today"})
    body = r.json()
    assert r.status_code == 200
    assert body["label"] == "joy" and body["model_version"] == "3"
    assert set(body["scores"]) == {"joy", "sadness", "anger"}


def test_empty_text_rejected(client):
    assert client.post("/predict", json={"text": ""}).status_code == 422


def test_no_model_returns_503():
    serve.STATE["clf"] = None
    assert TestClient(serve.app).get("/health").status_code == 503
