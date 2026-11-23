"""
predict_tools.py
----------------
Everything the dashboard needs to make and explain a prediction for ONE incident.

    make_row()    turn simple form inputs into the exact columns the models expect
    predict()     chance of a major delay + expected delay in minutes
    explain()     how much each group of inputs (incident, route, time, weather, events)
                  moved the prediction away from a typical incident (Shapley values)
    what_if()     change one input step by step and watch the prediction move
"""

import datetime as dt
import itertools
import math

import holidays
import pandas as pd

from config import INCIDENT_FEATURES
from features import RUSH_HOURS

# A "typical" incident: the most common values in the data (weekday noon, mild, dry)
TYPICAL = {
    "incident": "Mechanical", "route": "32", "direction": "N", "at_station": 0,
    "date": dt.date(2024, 5, 15), "hour": 12, "game": 0,
    "weather": "Cloudy", "temperature_c": 10.0, "precipitation_mm": 0.0, "snowfall_cm": 0.0,
    "snow_depth_cm": 0.0, "wind_kmh": 12.0, "gust_kmh": 25.0,
}

# Inputs grouped the way a person thinks about them
GROUPS = {
    "Incident type": ["incident"],
    "Route & place": ["route", "direction", "at_station"],
    "Time & day": ["date", "hour"],
    "Weather": ["weather", "temperature_c", "precipitation_mm", "snowfall_cm", "snow_depth_cm",
                "wind_kmh", "gust_kmh"],
    "Blue Jays game": ["game"],
}

_ON_HOLIDAYS = holidays.CA(subdiv="ON", years=range(2020, 2031))


def make_row(inputs: dict) -> dict:
    date = inputs["date"]
    hour = int(inputs["hour"])
    weekend = int(date.weekday() >= 5)
    holiday = int(date in _ON_HOLIDAYS)
    return {
        "incident": inputs["incident"], "route": str(inputs["route"]), "direction": inputs["direction"],
        "weather": inputs["weather"], "day_of_week": date.strftime("%A"),
        "hour": hour, "month": date.month, "is_weekend": weekend, "is_holiday": holiday,
        "is_rush_hour": int(hour in RUSH_HOURS and not weekend and not holiday),
        "at_station": int(inputs["at_station"]),
        "temperature_c": inputs["temperature_c"], "precipitation_mm": inputs["precipitation_mm"],
        "snowfall_cm": inputs["snowfall_cm"], "snow_depth_cm": inputs["snow_depth_cm"],
        "wind_kmh": inputs["wind_kmh"], "gust_kmh": inputs["gust_kmh"],
        "jays_game_window": int(inputs["game"]),
    }


def to_frame(list_of_inputs) -> pd.DataFrame:
    return pd.DataFrame([make_row(i) for i in list_of_inputs])[INCIDENT_FEATURES]


def predict(clf, reg, inputs: dict):
    X = to_frame([inputs])
    return float(clf.predict_proba(X)[0, 1]), max(float(reg.predict(X)[0]), 0.0)


def explain(reg, inputs: dict) -> pd.DataFrame:
    """
    Start from a typical incident and switch groups of inputs to "your incident".
    Each group's bar is its average effect over every order of switching (a Shapley
    value), so the bars add up exactly to the prediction.
    """
    names = list(GROUPS)
    combos = list(itertools.product([0, 1], repeat=len(names)))
    rows = []
    for combo in combos:
        mixed = dict(TYPICAL)
        for use_yours, group in zip(combo, names):
            if use_yours:
                for key in GROUPS[group]:
                    mixed[key] = inputs[key]
        rows.append(mixed)
    preds = dict(zip(combos, reg.predict(to_frame(rows))))

    n = len(names)
    out = [{"factor": "Typical incident", "minutes": float(preds[(0,) * n])}]
    for i, group in enumerate(names):
        total = 0.0
        for combo in combos:
            if combo[i]:
                continue
            k = sum(combo)
            weight = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            total += weight * (preds[combo[:i] + (1,) + combo[i + 1:]] - preds[combo])
        out.append({"factor": group, "minutes": float(total)})
    out.append({"factor": "Your incident", "minutes": float(preds[(1,) * n])})
    return pd.DataFrame(out)


def what_if(clf, reg, inputs: dict, key: str, values) -> pd.DataFrame:
    rows = [dict(inputs, **{key: v}) for v in values]
    X = to_frame(rows)
    return pd.DataFrame({key: list(values),
                         "major_chance": clf.predict_proba(X)[:, 1],
                         "expected_delay": reg.predict(X).clip(min=0)})
