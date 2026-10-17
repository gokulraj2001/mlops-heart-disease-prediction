"""Request / response schemas (input validation lives here)."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class PatientFeatures(BaseModel):
    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=50, le=260, description="Resting BP (mm Hg)")
    chol: float = Field(..., ge=80, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(..., ge=50, le=250, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression")
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST (1-3)")
    ca: Optional[int] = Field(None, ge=0, le=3, description="Major vessels (0-3); may be null")
    thal: Optional[int] = Field(None, description="3 normal, 6 fixed, 7 reversible; may be null")

    @field_validator("thal")
    @classmethod
    def _thal_values(cls, v):
        if v is not None and v not in (3, 6, 7):
            raise ValueError("thal must be one of 3, 6, 7 (or null)")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
                "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
                "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = heart disease predicted, 0 = not")
    label: str
    confidence: float = Field(..., description="Probability of the predicted class")
    probability_disease: float
    risk_band: str = Field(..., description="low / medium / high, from probability_disease")
    model_name: str

    model_config = {"protected_namespaces": ()}
