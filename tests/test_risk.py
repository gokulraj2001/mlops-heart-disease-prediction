import pytest

from api.risk import BANDS, risk_band
from src import config


@pytest.mark.parametrize("p,expected", [
    (0.0, "low"), (0.10, "low"), (0.3299, "low"),
    (config.LOW_RISK_MAX, "medium"), (0.5, "medium"), (0.6599, "medium"),
    (config.HIGH_RISK_MIN, "high"), (0.9, "high"), (1.0, "high"),
])
def test_band_boundaries(p, expected):
    assert risk_band(p) == expected


@pytest.mark.parametrize("p", [-0.01, 1.01])
def test_out_of_range_probability_rejected(p):
    with pytest.raises(ValueError):
        risk_band(p)


def test_thresholds_are_ordered():
    assert 0 < config.LOW_RISK_MAX < config.HIGH_RISK_MIN < 1
    assert BANDS == ("low", "medium", "high")
