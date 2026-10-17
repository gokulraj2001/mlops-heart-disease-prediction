#!/usr/bin/env bash
# Usage: scripts/smoke_test.sh [base_url]   (default http://localhost:8000)
set -euo pipefail
URL="${1:-http://localhost:8000}"
echo "== /health";  curl -fsS "$URL/health"; echo
echo "== /predict"
curl -fsS -X POST "$URL/predict" -H 'Content-Type: application/json' -d '{
  "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
  "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
  "ca": 0, "thal": 6}'
echo
echo "== invalid input must be rejected (422)"
code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$URL/predict" \
  -H 'Content-Type: application/json' -d '{"age": -1}')
[ "$code" = "422" ] && echo "OK ($code)" || { echo "expected 422, got $code"; exit 1; }
echo "== /metrics"; curl -fsS "$URL/metrics" | grep -m3 -E "^api_requests_total|^model_predictions_total"
echo "SMOKE TEST PASSED"
