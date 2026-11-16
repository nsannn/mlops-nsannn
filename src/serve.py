# src/serve.py
"""Inference API: loads the @champion model from the MLflow registry at start-up.

Run locally: uvicorn serve:app --app-dir src --port 8000   (needs MLFLOW_TRACKING_URI)
The image contains no model: MODEL_URI (default models:/tweet-emotion@champion) is resolved at start-up.
"""
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_URI = os.environ.get("MODEL_URI", "models:/tweet-emotion@champion")
STATE: dict = {"pipe": None, "info": {"model_uri": MODEL_URI}}
# One inference at a time: each call already uses all OMP_NUM_THREADS cores, so parallel calls would only
# oversubscribe the pod's CPU limit and get throttled. Requests queue here instead.
INFERENCE_LOCK = threading.Lock()


def load_champion():
    """Resolve the alias to a version, then load exactly that version through MLflow."""
    import mlflow  # heavy import; unit tests never reach it

    name, alias = MODEL_URI.removeprefix("models:/").split("@")
    mv = mlflow.MlflowClient().get_model_version_by_alias(name, alias)
    pipe = mlflow.transformers.load_model(f"models:/{name}/{mv.version}")
    return pipe, {"model_uri": MODEL_URI, "model": name, "version": mv.version, "run_id": mv.run_id}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.environ.get("SKIP_MODEL_LOAD") != "1":  # unit tests inject a fake model instead
        STATE["pipe"], STATE["info"] = load_champion()
    yield


app = FastAPI(title="Tweet emotion API", lifespan=lifespan)


class PredictIn(BaseModel):
    text: str = Field(min_length=1, max_length=500)


@app.get("/health")
def health():
    if STATE["pipe"] is None:
        raise HTTPException(status_code=503, detail={"status": "no model", **STATE["info"]})
    return {"status": "ok", **STATE["info"]}


@app.post("/predict")
def predict(req: PredictIn):
    if STATE["pipe"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    t0 = time.perf_counter()
    with INFERENCE_LOCK:
        raw = STATE["pipe"](req.text, top_k=None)
    scores = {s["label"]: round(s["score"], 4) for s in raw}
    label = max(scores, key=scores.get)
    return {"label": label, "score": scores[label], "scores": scores,
            "model_version": STATE["info"]["version"],
            "inference_ms": round((time.perf_counter() - t0) * 1000, 2)}
