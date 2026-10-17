"""Map a predicted disease probability to a coarse risk band."""
from src import config

BANDS = ("low", "medium", "high")


def risk_band(probability: float) -> str:
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}")
    if probability < config.LOW_RISK_MAX:
        return "low"
    if probability < config.HIGH_RISK_MIN:
        return "medium"
    return "high"
