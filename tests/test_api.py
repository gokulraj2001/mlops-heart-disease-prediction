import pytest
from fastapi.testclient import TestClient

GOOD = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
        "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
        "ca": 0, "thal": 6}


@pytest.fixture()
def client(model_dir, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_dir / "model.joblib"))
    from api.app import app
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_returns_prediction_and_confidence(client):
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert 0 <= body["probability_disease"] <= 1
    assert body["model_name"] == "test_lr"
    assert body["risk_band"] in ("low", "medium", "high")


def test_predict_accepts_null_for_ca_and_thal(client):
    r = client.post("/predict", json={**GOOD, "ca": None, "thal": None})
    assert r.status_code == 200


def test_predict_accepts_missing_optional_fields(client):
    payload = {k: v for k, v in GOOD.items() if k not in ("ca", "thal")}
    assert client.post("/predict", json=payload).status_code == 200


@pytest.mark.parametrize("field,value", [
    ("age", -5), ("sex", 2), ("cp", 9), ("thal", 5), ("chol", "abc"), ("oldpeak", 50),
])
def test_predict_rejects_invalid_values(client, field, value):
    r = client.post("/predict", json={**GOOD, field: value})
    assert r.status_code == 422


def test_predict_rejects_missing_required_field(client):
    bad = {k: v for k, v in GOOD.items() if k != "age"}
    assert client.post("/predict", json=bad).status_code == 422


def test_metrics_endpoint_counts_requests(client):
    client.post("/predict", json=GOOD)
    text = client.get("/metrics").text
    assert "api_requests_total" in text
    assert 'endpoint="/predict"' in text
    assert "model_predictions_total" in text
    assert "model_risk_band_total" in text
    assert "api_request_latency_seconds_bucket" in text


def test_health_503_when_model_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    from api.app import app
    with TestClient(app) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/predict", json=GOOD).status_code == 503
