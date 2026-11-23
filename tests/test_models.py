"""Smoke tests: the saved models load and give sensible predictions; the app starts."""

import datetime as dt
from pathlib import Path

import joblib
import pytest

from config import MODELS_DIR
from predict_tools import TYPICAL, explain, predict, what_if

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(not (MODELS_DIR / "delay_regressor.joblib").exists(),
                                reason="models not trained yet")


@pytest.fixture(scope="module")
def models():
    return (joblib.load(MODELS_DIR / "major_delay_classifier.joblib"),
            joblib.load(MODELS_DIR / "delay_regressor.joblib"))


def test_predictions_are_in_a_sensible_range(models):
    clf, reg = models
    chance, minutes = predict(clf, reg, TYPICAL)
    assert 0 <= chance <= 1
    assert 0 <= minutes <= 120


def test_explanation_adds_up_to_the_prediction(models):
    clf, reg = models
    inputs = dict(TYPICAL, incident="Diversion", hour=2, weather="Snow", snowfall_cm=1.0,
                  date=dt.date(2024, 12, 24))
    _, minutes = predict(clf, reg, inputs)
    e = explain(reg, inputs)
    assert abs(e["minutes"].iloc[:-1].sum() - e["minutes"].iloc[-1]) < 1e-6
    assert abs(e["minutes"].iloc[-1] - minutes) < 1e-6


def test_what_if_returns_one_row_per_value(models):
    clf, reg = models
    curve = what_if(clf, reg, TYPICAL, "hour", list(range(24)))
    assert len(curve) == 24


def test_app_starts_without_errors():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300).run()
    assert not at.exception
