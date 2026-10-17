# Heart Disease Risk - End-to-End MLOps

Predicts heart-disease risk from the UCI Heart Disease (Cleveland) dataset.

## Status
- [x] 1. Data acquisition and EDA
- [x] 2. Feature engineering and model development
- [x] 3. MLflow experiment tracking
- [x] 4. Packaging and reproducibility
- [x] 5. Tests (pytest) and GitHub Actions CI
- [x] 6. FastAPI service + Dockerfile
- [x] 7. Kubernetes manifests
- [x] 8. Logging + Prometheus/Grafana
- [ ] 9. Report, screenshots, video (you produce these from real runs)

## Setup (Python 3.10-3.12)
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the pipeline
```bash
python scripts/download_data.py     # -> data/raw/heart_disease_raw.csv
python -m src.eda                   # cleans data -> data/processed/, figures -> reports_out/eda/
python -m src.train                 # tunes LR + RF + GB, logs to MLflow, saves models/model.joblib
mlflow ui --backend-store-uri ./mlruns   # browse runs at http://127.0.0.1:5000
```
Use `python -m src.train --quick` for a fast run with small grids.

## Design notes
- Target `num` (0-4) is binarised: 0 = no disease, 1-4 = disease.
- Missing values (`ca`, `thal`) are imputed *inside* the sklearn Pipeline, so they are fit on training folds only (no leakage) and applied identically when serving.
- The saved artifact is the full Pipeline (imputer, scaler, one-hot encoder, classifier), so training and serving cannot drift apart.
- Model selection: GridSearchCV with stratified 5-fold CV on ROC-AUC; the winner is scored once on a held-out 20% test set.

## Test, lint
```bash
flake8 .
pytest -v --cov=src --cov=api
```

## Serve locally
```bash
uvicorn api.app:app --port 8000        # docs at http://localhost:8000/docs
./scripts/smoke_test.sh                # health, predict, validation, metrics
```
Example request:
```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```
The response includes `risk_band` (low < 0.33 <= medium < 0.66 <= high, from `probability_disease`; illustrative cut-offs, not clinically validated). `ca` and `thal` may be `null` or omitted; the pipeline imputes them. Raw patient inputs are not logged unless `LOG_INPUTS=1`.

## Docker (requires models/model.joblib from training)
```bash
docker build -t heart-disease-api:latest .
docker run -p 8000:8000 heart-disease-api:latest
./scripts/smoke_test.sh
```

## Kubernetes (Minikube or Docker Desktop)
```bash
# Minikube: build the image inside the cluster's Docker daemon
eval $(minikube docker-env) && docker build -t heart-disease-api:latest .
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl get pods,svc
minikube tunnel                         # separate terminal; Docker Desktop needs no tunnel
./scripts/smoke_test.sh http://localhost   # or the EXTERNAL-IP shown by `kubectl get svc`
```

## Monitoring
Local (Docker Compose): `docker compose -f monitoring/docker-compose.yml up --build`
- API http://localhost:8000, Prometheus http://localhost:9090, Grafana http://localhost:3000 (admin/admin, dashboard "Heart Disease API" is pre-provisioned).

In Kubernetes: `kubectl apply -f k8s/monitoring.yaml`, then
`kubectl port-forward svc/prometheus 9090` and `kubectl port-forward svc/grafana 3000`; import `monitoring/grafana/dashboards/heart_api.json` in Grafana.

## Notes
- `models/model.joblib` is a pickle: the serving image pins the same scikit-learn/numpy/pandas as `requirements.txt`. Retrain and rebuild together.
