"""Single source of truth for paths, column groups and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "heart_disease_raw.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "heart_disease_clean.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
PLOTS_DIR = ROOT / "reports_out"

TARGET_RAW = "num"      # 0-4 severity in the original data
TARGET = "target"       # binarised: 0 = no disease, 1 = disease

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
# Categorical / ordinal-coded features -> imputed (mode) then one-hot encoded
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES  # raw inputs (what the API accepts)

# Engineered inside the pipeline, never supplied by the caller:
#   hr_reserve = (220 - age) - thalach  -> beats/min below the age-predicted max HR
DERIVED_FEATURES = ["hr_reserve"]

RANDOM_STATE = 42

# Risk bands applied to the predicted probability of heart disease.
# Illustrative cut-offs for triage-style reporting, NOT clinically validated.
LOW_RISK_MAX = 0.33     # p <  0.33          -> "low"
HIGH_RISK_MIN = 0.66    # p >= 0.66          -> "high"; in between -> "medium"
