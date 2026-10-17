"""FastAPI service for heart-disease risk predictions.

Endpoints: POST /predict, GET /health, GET /metrics, GET / (docs at /docs).
Config (env): MODEL_PATH (default models/model.joblib), LOG_LEVEL, LOG_INPUTS=1
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api import metrics
from api.risk import risk_band
from api.schemas import PatientFeatures, PredictionResponse
from src import config

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("heart-api")

STATE: dict = {"model": None, "name": "unknown"}


def load_model() -> None:
    STATE.update(model=None, name="unknown")  # never serve a stale model
    path = Path(os.getenv("MODEL_PATH", str(config.MODEL_PATH)))
    STATE["model"] = joblib.load(path)
    meta = path.with_name("metrics.json")
    if meta.exists():
        STATE["name"] = json.loads(meta.read_text()).get("selected_model", "unknown")
    metrics.MODEL_LOADED.set(1)
    log.info("model loaded from %s (%s)", path, STATE["name"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
    except Exception:  # keep serving /health so k8s can see the failure
        metrics.MODEL_LOADED.set(0)
        log.exception("could not load model")
    yield


app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def observe(request: Request, call_next):
    start = time.perf_counter()
    metrics.IN_FLIGHT.inc()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - start
        metrics.IN_FLIGHT.dec()
        route = request.scope.get("route")
        endpoint = route.path if route else "unmatched"
        metrics.REQUESTS.labels(request.method, endpoint, str(status)).inc()
        metrics.LATENCY.labels(endpoint).observe(elapsed)
        if endpoint != "/metrics":
            log.info("request method=%s path=%s status=%s latency_ms=%.1f client=%s",
                     request.method, request.url.path, status, elapsed * 1000,
                     request.client.host if request.client else "-")


@app.get("/")
def root():
    return {"service": "heart-disease-risk-api", "docs": "/docs"}


@app.get("/health")
def health():
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model": STATE["name"]}


@app.get("/metrics")
def prom_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures):
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    row = patient.model_dump()
    X = pd.DataFrame([row], columns=config.FEATURES).astype("float64")
    proba = float(STATE["model"].predict_proba(X)[0, 1])
    pred = int(proba >= 0.5)
    confidence = proba if pred == 1 else 1.0 - proba
    label = "heart disease likely" if pred else "heart disease unlikely"

    band = risk_band(proba)
    metrics.PREDICTIONS.labels(str(pred)).inc()
    metrics.RISK_BANDS.labels(band).inc()
    metrics.CONFIDENCE.observe(confidence)
    # Raw patient data is only logged if explicitly enabled (privacy default)
    extra = f" inputs={json.dumps(row)}" if os.getenv("LOG_INPUTS") == "1" else ""
    log.info("prediction=%d probability=%.4f risk_band=%s confidence=%.4f%s",
             pred, proba, band, confidence, extra)
    return PredictionResponse(prediction=pred, label=label,
                              confidence=round(confidence, 4),
                              probability_disease=round(proba, 4),
                              risk_band=band,
                              model_name=STATE["name"])
