# src/serve_onnx.py
"""Slim inference API (U2-E4): ONNX Runtime + tokenizers, no torch or transformers.

Resolves MODEL_URI (default models:/tweet-emotion-onnx@champion) to a version at start-up, downloads that
version's files from MLflow and serves them. Same /health and /predict contract as serve.py.
Run locally: uvicorn serve_onnx:app --app-dir src --port 8001   (needs MLFLOW_TRACKING_URI)
"""
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_URI = os.environ.get("MODEL_URI", "models:/tweet-emotion-onnx@champion")
STATE: dict = {"clf": None, "info": {"model_uri": MODEL_URI}}
# One inference at a time: each call already uses all OMP_NUM_THREADS cores.
INFERENCE_LOCK = threading.Lock()


def load_champion():
    """Resolve the alias to a version, download exactly that version's files, build the classifier."""
    import mlflow  # mlflow-skinny: registry and artifact client only

    from onnx_infer import OnnxClassifier

    name, alias = MODEL_URI.removeprefix("models:/").split("@")
    mv = mlflow.MlflowClient().get_model_version_by_alias(name, alias)
    local = mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{name}/{mv.version}", dst_path=tempfile.mkdtemp())
    clf = OnnxClassifier(Path(local) / "artifacts")
    return clf, {"model_uri": MODEL_URI, "model": name, "version": mv.version, "run_id": mv.run_id,
                 "runtime": "onnxruntime"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.environ.get("SKIP_MODEL_LOAD") != "1":  # unit tests inject a fake classifier instead
        STATE["clf"], STATE["info"] = load_champion()
    yield


app = FastAPI(title="Tweet emotion API (ONNX)", lifespan=lifespan)


class PredictIn(BaseModel):
    text: str = Field(min_length=1, max_length=500)


@app.get("/health")
def health():
    if STATE["clf"] is None:
        raise HTTPException(status_code=503, detail={"status": "no model", **STATE["info"]})
    return {"status": "ok", **STATE["info"]}


@app.post("/predict")
def predict(req: PredictIn):
    if STATE["clf"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    t0 = time.perf_counter()
    with INFERENCE_LOCK:
        raw = STATE["clf"].scores(req.text)
    scores = {label: round(s, 4) for label, s in raw.items()}
    label = max(scores, key=scores.get)
    return {"label": label, "score": scores[label], "scores": scores,
            "model_version": STATE["info"]["version"],
            "inference_ms": round((time.perf_counter() - t0) * 1000, 2)}
