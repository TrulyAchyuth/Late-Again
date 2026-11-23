"""
predict_tools.py
----------------
Helper functions the dashboard uses to make predictions:

    make_trip()      turn simple user inputs into the exact columns the model needs
    predict()        chance of delay + expected minutes for one trip
    explain()        "why this prediction?" - how much each group of factors adds
    what_if()        change ONE input step by step and watch the prediction move
    plan_routes()    compare every route under the same conditions

All of these only call the trained models. They do not change them.
"""

import datetime as dt
import itertools
import math

import pandas as pd

from config import FEATURES

PEAK_HOURS = (7, 8, 9, 17, 18, 19)

# A "typical calm trip" used as the reference point in explain().
# Weekday midday, clear weather, no event, normal traffic.
TYPICAL = {
    "weather_condition": "Clear", "temperature_c": 15.0, "humidity_pct": 60.0,
    "wind_speed_kmh": 12.0, "precipitation_mm": 0.0,
    "event_type": "No Event", "event_attendance": 0,
    "traffic_congestion_index": 3.0,
    "hour": 12, "weekday_offset": 0, "is_holiday": False,
}

# Which inputs belong to which group in the explanation
GROUPS = {
    "Weather": ["weather_condition", "temperature_c", "humidity_pct", "wind_speed_kmh", "precipitation_mm"],
    "City event": ["event_type", "event_attendance"],
    "Traffic": ["traffic_congestion_index"],
    "Time & day": ["hour", "date", "is_holiday"],
}


def season_of(month: int) -> str:
    return {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer"}.get(month, "Autumn")


def make_trip(inputs: dict, routes: pd.DataFrame) -> pd.DataFrame:
    """
    inputs: dict with route_id, date (datetime.date), hour, is_holiday, weather values,
            event_type, event_attendance, traffic_congestion_index
    routes: DataFrame indexed by route_id with transport_type, distance_km, scheduled_duration_min
    Returns a one-row DataFrame with the model's feature columns.
    """
    route = routes.loc[inputs["route_id"]]
    date = inputs["date"]
    is_weekend = int(date.weekday() >= 5)
    is_holiday = int(bool(inputs["is_holiday"]))
    no_event = inputs["event_type"] == "No Event"
    row = {
        "transport_type": route["transport_type"],
        "route_id": inputs["route_id"],
        "weather_condition": inputs["weather_condition"],
        "event_type": inputs["event_type"],
        "season": season_of(date.month),
        "day_of_week": date.strftime("%A"),
        "hour": int(inputs["hour"]),
        "month": date.month,
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "is_peak_hour": int(inputs["hour"] in PEAK_HOURS and not is_weekend and not is_holiday),
        "distance_km": route["distance_km"],
        "scheduled_duration_min": route["scheduled_duration_min"],
        "temperature_c": inputs["temperature_c"],
        "humidity_pct": inputs["humidity_pct"],
        "wind_speed_kmh": inputs["wind_speed_kmh"],
        "precipitation_mm": inputs["precipitation_mm"],
        "event_attendance": 0 if no_event else inputs["event_attendance"],
        "traffic_congestion_index": inputs["traffic_congestion_index"],
    }
    return pd.DataFrame([row])[FEATURES]


def predict(clf, reg, inputs, routes):
    """Return (probability of delay, expected delay in minutes)."""
    trip = make_trip(inputs, routes)
    return float(clf.predict_proba(trip)[0, 1]), float(reg.predict(trip)[0])


def _typical_inputs(inputs: dict) -> dict:
    """Same route and month, but everything else set to a typical calm trip."""
    calm = dict(inputs)
    for key in ["weather_condition", "temperature_c", "humidity_pct", "wind_speed_kmh",
                "precipitation_mm", "event_type", "event_attendance",
                "traffic_congestion_index", "hour", "is_holiday"]:
        calm[key] = TYPICAL[key]
    # Move the date to a Wednesday in the same week, so it is a normal weekday
    d = inputs["date"]
    calm["date"] = d + dt.timedelta(days=2 - d.weekday())
    return calm


def explain(reg, inputs: dict, routes) -> pd.DataFrame:
    """
    "Why this prediction?" for the expected delay in minutes.

    1. Start from the same route on a typical calm day (the reference).
    2. Switch groups of factors (weather, event, traffic, time) from "typical"
       to "your trip" and see how the predicted delay changes.
    3. The order we switch them in matters a little, so we try ALL orders and
       average (this fair-sharing idea is called a Shapley value).
    The contributions add up exactly: reference + contributions = prediction.
    """
    calm = _typical_inputs(inputs)
    groups = list(GROUPS)

    # Predict every combination of "typical" vs "your trip" groups (2^4 = 16 rows)
    combos = list(itertools.product([0, 1], repeat=len(groups)))
    rows = []
    for combo in combos:
        mixed = dict(calm)
        for use_trip, group in zip(combo, groups):
            if use_trip:
                for k in GROUPS[group]:
                    mixed[k] = inputs[k]
        rows.append(make_trip(mixed, routes))
    preds = dict(zip(combos, reg.predict(pd.concat(rows, ignore_index=True))))

    n = len(groups)
    result = [{"factor": "Route on a typical calm day", "minutes": float(preds[(0,) * n])}]
    for i, group in enumerate(groups):
        total = 0.0
        for combo in combos:
            if combo[i] == 1:
                continue
            with_i = combo[:i] + (1,) + combo[i + 1:]
            k = sum(combo)  # how many other groups are already "your trip"
            weight = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            total += weight * (preds[with_i] - preds[combo])
        result.append({"factor": group, "minutes": float(total)})
    result.append({"factor": "Predicted delay", "minutes": float(preds[(1,) * n])})
    return pd.DataFrame(result)


def what_if(clf, reg, inputs: dict, routes, feature: str, values) -> pd.DataFrame:
    """Predict the trip again for each value of one input, keeping the rest fixed."""
    rows = []
    for v in values:
        changed = dict(inputs)
        changed[feature] = v
        prob, minutes = predict(clf, reg, changed, routes)
        rows.append({feature: v, "chance_of_delay": prob, "expected_delay_min": minutes})
    return pd.DataFrame(rows)


def plan_routes(clf, reg, inputs: dict, routes, route_ids=None) -> pd.DataFrame:
    """Same conditions, every route: which one is likely to be least delayed?"""
    rows = []
    for rid in (route_ids if route_ids is not None else routes.index):
        changed = dict(inputs, route_id=rid)
        prob, minutes = predict(clf, reg, changed, routes)
        sched = float(routes.loc[rid, "scheduled_duration_min"])
        rows.append({
            "route": rid,
            "type": routes.loc[rid, "transport_type"],
            "distance_km": routes.loc[rid, "distance_km"],
            "scheduled_min": round(sched),
            "chance_of_delay_%": round(prob * 100),
            "expected_delay_min": round(minutes, 1),
            "expected_total_min": round(sched + max(minutes, 0), 1),
        })
    return pd.DataFrame(rows).sort_values("expected_delay_min").reset_index(drop=True)
