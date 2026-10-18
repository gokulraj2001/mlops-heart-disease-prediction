#!/usr/bin/env bash
# One-stop runner for the lab. Every stage prints to the screen AND saves its
# output to lab_outputs/<stage>.txt, so you never need to type `| tee`.
#
#   bash scripts/lab.sh <stage>
#
# Stages (run in this order):
#   env      check what the lab provides
#   setup    create .venv and install requirements.txt (clean-setup proof)
#   data     download dataset, run EDA, copy figures to screenshots/
#   train    final model + no-hr_reserve baseline, saves both metrics.json
#   test     flake8 + pytest with coverage
#   api      start the API, run the smoke test, stop it
#   docker   build image, run container, smoke test, show logs
#   monitor  start API + Prometheus + Grafana, generate traffic (leaves it running)
#   monitor-down   stop the monitoring stack
#   k8s      deploy to the cluster, verify through the service
#   collect  pack lab_outputs/ and screenshots/ into one file to take away
#   core     env + setup + data + train + test + api in one go
#
# If docker needs sudo:  DOCKER="sudo docker" bash scripts/lab.sh docker
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=lab_outputs
PY=.venv/bin/python
DOCKER=${DOCKER:-docker}
mkdir -p "$OUT" screenshots

LOW='{"age":35,"sex":0,"cp":3,"trestbps":115,"chol":190,"fbs":0,"restecg":0,"thalach":185,"exang":0,"oldpeak":0.0,"slope":1,"ca":0,"thal":3}'
MID='{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
HIGH='{"age":66,"sex":1,"cp":4,"trestbps":160,"chol":286,"fbs":0,"restecg":2,"thalach":108,"exang":1,"oldpeak":1.5,"slope":2,"ca":3,"thal":3}'

say() { echo; echo "=== $* ==="; }
need_venv() { [ -x "$PY" ] || { echo "No .venv yet - run: bash scripts/lab.sh setup"; return 1; }; }
need_model() { [ -f models/model.joblib ] || { echo "No models/model.joblib - run: bash scripts/lab.sh train"; return 1; }; }

wait_health() {   # wait_health <base_url> [seconds]
  local url=$1 n=${2:-60} i
  for i in $(seq 1 "$n"); do
    curl -fs "$url/health" >/dev/null 2>&1 && { echo "API healthy after ${i}s"; return 0; }
    sleep 1
  done
  echo "API did not become healthy within ${n}s"; return 1
}

compose() {
  if $DOCKER compose version >/dev/null 2>&1; then $DOCKER compose "$@"; else docker-compose "$@"; fi
}

stage_env() {
  say "Environment"
  echo "host: $(hostname)"; echo "date: $(date)"; echo "user: $(whoami)"
  uname -sr
  say "Tools (NOT FOUND is fine to report - see fallbacks)"
  local t
  for t in python3 git curl; do
    if command -v "$t" >/dev/null; then printf '%-9s %s\n' "$t" "$("$t" --version 2>&1 | head -1)"
    else printf '%-9s NOT FOUND\n' "$t"; fi
  done
  if command -v "${DOCKER%% *}" >/dev/null; then printf '%-9s %s\n' docker "$($DOCKER --version 2>&1 | head -1)"
  else printf '%-9s NOT FOUND\n' docker; fi
  if command -v kubectl >/dev/null; then printf '%-9s %s\n' kubectl "$(kubectl version --client 2>&1 | head -1)"
  else printf '%-9s NOT FOUND\n' kubectl; fi
  if command -v minikube >/dev/null; then printf '%-9s %s\n' minikube "$(minikube version 2>&1 | head -1)"
  else printf '%-9s NOT FOUND\n' minikube; fi
  say "Internet checks"
  local u
  for u in https://pypi.org https://archive.ics.uci.edu https://registry-1.docker.io https://github.com; do
    printf '%-34s ' "$u"; curl -s -o /dev/null -m 8 -w "HTTP %{http_code}\n" "$u" || true
  done
  say "Docker daemon / cluster"
  $DOCKER ps >/dev/null 2>&1 && echo "docker daemon: reachable" || echo "docker daemon: NOT reachable (try DOCKER=\"sudo docker\")"
  kubectl cluster-info >/dev/null 2>&1 && echo "kubernetes: cluster reachable" || echo "kubernetes: no cluster reachable yet"
  return 0
}

stage_setup() {
  say "Create virtualenv and install requirements"
  local t0=$SECONDS
  [ -x "$PY" ] || python3 -m venv .venv || return 1
  .venv/bin/pip install -r requirements.txt || return 1
  echo "Install finished in $((SECONDS - t0))s"
  .venv/bin/pip freeze > "$OUT/pip_freeze.txt"
  "$PY" --version
}

stage_data() {
  need_venv || return 1
  say "Download dataset"
  "$PY" scripts/download_data.py || {
    [ -f data/raw/heart_disease_raw.csv ] && echo "[info] download failed; using the dataset committed in the repo" \
      || { echo "No dataset available"; return 1; }; }
  say "EDA"
  "$PY" -m src.eda || return 1
  cp reports_out/eda/*.png screenshots/ && echo "Copied EDA figures to screenshots/" && ls screenshots
}

stage_train() {
  need_venv || return 1
  say "Final model (with hr_reserve)"
  "$PY" -m src.train || return 1
  cp models/metrics.json "$OUT/metrics_final.json"
  say "Baseline (without hr_reserve)"
  "$PY" -m src.train --without-hr-reserve --experiment baseline-no-hrr \
        --model-dir models_baseline || return 1
  cp models_baseline/metrics.json "$OUT/metrics_baseline.json"
  say "Summary"
  "$PY" - <<'PYEOF'
import json
for tag in ("final", "baseline"):
    m = json.load(open(f"lab_outputs/metrics_{tag}.json"))
    print(f"[{tag}] selected={m['selected_model']}  derived={m.get('derived_features')}")
    for k, v in m["all_models"].items():
        print(f"    {k:20s} cv_roc_auc={v['cv_roc_auc_mean']:.4f}  test_roc_auc={v['test_roc_auc']:.4f}")
    t = m["test_metrics"]
    print("    selected test:", {k: round(x, 4) for k, x in t.items()})
PYEOF
  echo; echo "To browse runs:  .venv/bin/mlflow ui --backend-store-uri ./mlruns   (then open http://127.0.0.1:5000)"
}

stage_test() {
  need_venv || return 1
  say "flake8"; .venv/bin/flake8 . || return 1; echo "flake8 clean"
  say "pytest"; .venv/bin/pytest -v --cov=src --cov=api --cov-report=term-missing
}

stage_api() {
  need_venv || return 1; need_model || return 1
  say "Start API on :8000"
  .venv/bin/uvicorn api.app:app --port 8000 > "$OUT/api_server.log" 2>&1 &
  local pid=$!
  if ! wait_health http://localhost:8000; then kill "$pid" 2>/dev/null; tail -20 "$OUT/api_server.log"; return 1; fi
  say "Smoke test"
  bash scripts/smoke_test.sh http://localhost:8000; local rc=$?
  say "Server log (last lines)"; grep -E "request|prediction" "$OUT/api_server.log" | tail -6
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  return $rc
}

stage_docker() {
  need_model || return 1
  say "docker build"; $DOCKER build -t heart-disease-api:latest . || return 1
  say "image size"; $DOCKER images heart-disease-api
  $DOCKER rm -f api >/dev/null 2>&1
  say "docker run"; $DOCKER run -d --name api -p 8000:8000 heart-disease-api:latest || return 1
  wait_health http://localhost:8000 || { $DOCKER logs api | tail -30; $DOCKER rm -f api; return 1; }
  say "Smoke test against the container"
  bash scripts/smoke_test.sh http://localhost:8000; local rc=$?
  say "docker ps"; $DOCKER ps --filter name=api
  say "container logs"; $DOCKER logs api 2>&1 | tee "$OUT/docker_container.log" | tail -15
  $DOCKER rm -f api >/dev/null
  return $rc
}

stage_monitor() {
  need_model || return 1
  $DOCKER rm -f api >/dev/null 2>&1
  say "Start API + Prometheus + Grafana"
  compose -f monitoring/docker-compose.yml up -d --build || return 1
  wait_health http://localhost:8000 90 || { compose -f monitoring/docker-compose.yml logs api | tail -30; return 1; }
  say "Generating traffic (150 requests, mixed risk levels + a few invalid)"
  local i
  for i in $(seq 1 150); do
    case $((i % 3)) in 0) body=$LOW;; 1) body=$MID;; *) body=$HIGH;; esac
    [ $((i % 25)) -eq 0 ] && body='{"age":-1}'
    curl -s -o /dev/null -X POST localhost:8000/predict -H 'Content-Type: application/json' -d "$body"
    sleep 0.1
  done
  echo "Done."
  cat <<MSG

Open in a browser, take screenshots, then run:  bash scripts/lab.sh monitor-down
  Grafana     http://localhost:3000   (admin / admin) -> Dashboards -> "Heart Disease API"
  Prometheus  http://localhost:9090/targets   (state should be UP)
  API docs    http://localhost:8000/docs
Re-run this stage any time to generate more traffic so the graphs have fresh data.
MSG
}

stage_monitor_down() { compose -f monitoring/docker-compose.yml down; }

stage_k8s() {
  need_model || return 1
  command -v kubectl >/dev/null || { echo "kubectl not found"; return 1; }
  if ! kubectl cluster-info >/dev/null 2>&1; then
    command -v minikube >/dev/null || { echo "No cluster reachable and no minikube"; return 1; }
    say "minikube start"; minikube start --driver=docker || return 1
  fi
  say "Build image where the cluster can see it"
  if [ "$(kubectl config current-context)" = "minikube" ]; then
    eval "$(minikube docker-env)"; echo "(building inside minikube's docker)"
  fi
  $DOCKER build -t heart-disease-api:latest . || return 1
  say "Apply manifests"
  kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml || return 1
  kubectl rollout status deployment/heart-disease-api --timeout=180s || {
    kubectl get pods; kubectl describe pods | tail -30; return 1; }
  say "kubectl get pods,svc"; kubectl get pods,svc
  kubectl describe deployment heart-disease-api > "$OUT/k8s_describe.txt"
  say "Reach the service (port-forward; use 'minikube tunnel' in another terminal for an EXTERNAL-IP)"
  kubectl port-forward svc/heart-disease-api 8080:80 >"$OUT/k8s_portforward.log" 2>&1 &
  local pf=$!
  wait_health http://localhost:8080 30 || { kill "$pf" 2>/dev/null; return 1; }
  bash scripts/smoke_test.sh http://localhost:8080; local rc=$?
  kill "$pf" 2>/dev/null; wait "$pf" 2>/dev/null
  return $rc
}

stage_collect() {
  say "Packing evidence"
  ls -la "$OUT" screenshots
  if command -v zip >/dev/null; then zip -qr lab_evidence.zip "$OUT" screenshots && echo "Created lab_evidence.zip"
  else tar czf lab_evidence.tar.gz "$OUT" screenshots && echo "Created lab_evidence.tar.gz"; fi
  echo "Upload that one file (browser / email / drive). Screenshots you take yourself go in screenshots/ first."
}

run() {   # run <stage-name> <function>
  local name=$1 fn=$2
  "$fn" 2>&1 | tee "$OUT/${name}.txt"
  local rc=${PIPESTATUS[0]}
  echo; [ "$rc" -eq 0 ] && echo "[$name] OK  (saved to $OUT/${name}.txt)" || echo "[$name] FAILED (exit $rc) - see $OUT/${name}.txt"
  return "$rc"
}

case "${1:-}" in
  env)          run 00_env stage_env ;;
  setup)        run 01_setup stage_setup ;;
  data)         run 02_data_eda stage_data ;;
  train)        run 03_train stage_train ;;
  test)         run 04_tests stage_test ;;
  api)          run 05_api stage_api ;;
  docker)       run 06_docker stage_docker ;;
  monitor)      run 07_monitor stage_monitor ;;
  monitor-down) stage_monitor_down ;;
  k8s)          run 08_k8s stage_k8s ;;
  collect)      stage_collect ;;
  core)         for s in env setup data train test api; do bash "$0" "$s" || exit 1; done ;;
  *)            sed -n '2,24p' "$0"; exit 1 ;;
esac
