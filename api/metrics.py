"""Prometheus metrics exposed at /metrics."""
from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("api_requests_total", "HTTP requests",
                   ["method", "endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency",
                    ["endpoint"],
                    buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2.5))
PREDICTIONS = Counter("model_predictions_total", "Predictions by class",
                      ["label"])
RISK_BANDS = Counter("model_risk_band_total", "Predictions by risk band",
                     ["band"])
CONFIDENCE = Histogram("model_prediction_confidence", "Confidence of predictions",
                       buckets=(.5, .6, .7, .8, .9, .95, 1.0))
IN_FLIGHT = Gauge("api_requests_in_flight", "Requests currently being served")
MODEL_LOADED = Gauge("model_loaded", "1 if the model is loaded")
