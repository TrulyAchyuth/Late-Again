"""
generate_data.py
----------------
Creates a SYNTHETIC (made-up) dataset of bus, train, tram and metro trips.

Why synthetic?
    The public Kaggle dataset "Public Transport Delays with Weather & Events"
    is itself synthetic and needs a Kaggle login to download. So this script
    builds a similar dataset with the same kind of columns. The rules below are
    simple assumptions (e.g. "storms add delay"), NOT measured real-world facts.
    Any pattern you later "discover" in this data exists because we put it here.

Run:
    python src/generate_data.py
"""

import numpy as np
import pandas as pd

from config import RAW_DATA, RANDOM_STATE, DELAY_THRESHOLD_MIN

N_TRIPS = 6000
rng = np.random.default_rng(RANDOM_STATE)

# ---------------------------------------------------------------------------
# 1. Routes: each route has a transport type, length and two end stations
# ---------------------------------------------------------------------------
ROUTE_TYPES = ["Bus"] * 8 + ["Train"] * 5 + ["Tram"] * 4 + ["Metro"] * 3
DISTANCE_RANGE = {"Bus": (5, 20), "Train": (20, 80), "Tram": (4, 12), "Metro": (6, 25)}
AVG_SPEED_KMH = {"Bus": 20, "Train": 55, "Tram": 18, "Metro": 35}

routes = []
for i, t_type in enumerate(ROUTE_TYPES, start=1):
    lo, hi = DISTANCE_RANGE[t_type]
    stations = rng.choice(np.arange(1, 51), size=2, replace=False)
    routes.append({
        "route_id": f"Route_{i}",
        "transport_type": t_type,
        "distance_km": round(rng.uniform(lo, hi), 1),
        "station_a": f"Station_{stations[0]}",
        "station_b": f"Station_{stations[1]}",
    })
routes = pd.DataFrame(routes)

# ---------------------------------------------------------------------------
# 2. Calendar: one year of dates, with a made-up list of public holidays
# ---------------------------------------------------------------------------
dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
HOLIDAYS = pd.to_datetime([
    "2023-01-01", "2023-01-26", "2023-03-08", "2023-04-07", "2023-05-01",
    "2023-08-15", "2023-10-02", "2023-10-24", "2023-11-12", "2023-12-25",
])


def season_of(month: int) -> str:
    """Northern-hemisphere seasons."""
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


# ---------------------------------------------------------------------------
# 3. Daily weather: condition depends on the season
# ---------------------------------------------------------------------------
WEATHER_PROBS = {
    #            Clear Cloudy Fog  Rain HeavyRain Snow Storm
    "Winter": [0.25, 0.25, 0.12, 0.10, 0.04, 0.20, 0.04],
    "Spring": [0.35, 0.25, 0.08, 0.20, 0.07, 0.00, 0.05],
    "Summer": [0.45, 0.20, 0.02, 0.15, 0.10, 0.00, 0.08],
    "Autumn": [0.30, 0.28, 0.10, 0.20, 0.08, 0.00, 0.04],
}
CONDITIONS = ["Clear", "Cloudy", "Fog", "Rain", "Heavy Rain", "Snow", "Storm"]
BASE_TEMP = {"Winter": 2, "Spring": 14, "Summer": 26, "Autumn": 13}
PRECIP_RANGE = {"Clear": (0, 0), "Cloudy": (0, 0.5), "Fog": (0, 0.5), "Rain": (1, 8),
                "Heavy Rain": (8, 30), "Snow": (2, 15), "Storm": (10, 45)}

daily_weather = {}
for d in dates:
    s = season_of(d.month)
    cond = rng.choice(CONDITIONS, p=WEATHER_PROBS[s])
    lo, hi = PRECIP_RANGE[cond]
    wind = rng.gamma(2, 6) + (25 if cond == "Storm" else 0)
    daily_weather[d] = {
        "weather_condition": cond,
        "base_temp": BASE_TEMP[s] + rng.normal(0, 4) - (3 if cond == "Snow" else 0),
        "humidity_pct": float(np.clip(rng.normal(85 if lo > 0 else 60, 10), 20, 100)),
        "wind_speed_kmh": round(float(wind), 1),
        "precipitation_mm": round(float(rng.uniform(lo, hi)), 1),
    }

# ---------------------------------------------------------------------------
# 4. City events: about 1 in 4 days has an event during a time window
# ---------------------------------------------------------------------------
EVENT_INFO = {
    # type:      (attendance range, start hour, end hour, extra impact factor)
    "Concert":  ((5_000, 40_000), 17, 23, 1.0),
    "Sports":   ((10_000, 60_000), 16, 22, 1.1),
    "Festival": ((3_000, 30_000), 10, 21, 0.8),
    "Parade":   ((5_000, 25_000), 9, 14, 1.3),
    "Protest":  ((1_000, 15_000), 10, 17, 1.6),
}
daily_event = {}
for d in dates:
    if rng.random() < 0.25:
        e_type = rng.choice(list(EVENT_INFO))
        (lo, hi), start, end, _ = EVENT_INFO[e_type]
        daily_event[d] = (e_type, int(rng.integers(lo, hi)), start, end)

# ---------------------------------------------------------------------------
# 5. Trips
# ---------------------------------------------------------------------------
# Trips are more common at rush hour, so hours are sampled with weights
hours = np.arange(5, 24)
hour_w = np.array([1, 3, 6, 6, 4, 3, 3, 3, 3, 3, 3, 4, 6, 6, 4, 3, 2, 2, 1], dtype=float)
hour_w /= hour_w.sum()

# How strongly each mode is affected by weather / road traffic
WEATHER_SENS = {"Bus": 1.0, "Tram": 0.9, "Train": 0.8, "Metro": 0.4}
ROAD_SENS = {"Bus": 1.0, "Tram": 0.8, "Train": 0.2, "Metro": 0.1}
BASE_DELAY = {"Bus": 1.5, "Tram": 1.0, "Train": 0.5, "Metro": 0.0}
WEATHER_DELAY = {"Clear": 0, "Cloudy": 0.5, "Fog": 3, "Rain": 3,
                 "Heavy Rain": 7, "Snow": 9, "Storm": 12}
# Some routes are simply "worse" than others (old track, busy roads, ...)
route_effect = {r: rng.normal(0, 1.5) for r in routes["route_id"]}

rows = []
for i in range(N_TRIPS):
    route = routes.iloc[rng.integers(len(routes))]
    t_type = route["transport_type"]
    date = dates[rng.integers(len(dates))]
    hour = int(rng.choice(hours, p=hour_w))
    minute = int(rng.integers(0, 60))

    weekday = date.day_name()
    is_weekend = int(date.day_of_week >= 5)
    is_holiday = int(date in HOLIDAYS)
    is_peak = int((hour in (7, 8, 9) or hour in (17, 18, 19)) and not is_weekend and not is_holiday)

    w = daily_weather[date]
    temp = w["base_temp"] + (4 if 11 <= hour <= 16 else 0) + rng.normal(0, 1.5)

    # Event affects this trip only if the trip is during the event window
    e_type, attendance, e_factor = "No Event", 0, 0.0
    if date in daily_event:
        et, att, start, end = daily_event[date]
        if start <= hour <= end:
            e_type, attendance, e_factor = et, att, EVENT_INFO[et][3]

    # Traffic congestion index (0 = empty roads, 10 = gridlock)
    congestion = (3 + 2.5 * is_peak - 1.0 * is_weekend - 0.8 * is_holiday
                  + attendance / 15_000
                  + (1.5 if w["precipitation_mm"] > 5 else 0)
                  + rng.normal(0, 1))
    congestion = float(np.clip(congestion, 0, 10))

    # Scheduled times
    duration = route["distance_km"] / AVG_SPEED_KMH[t_type] * 60
    sched_dep = pd.Timestamp(date) + pd.Timedelta(hours=hour, minutes=minute)
    sched_arr = sched_dep + pd.Timedelta(minutes=round(duration))

    # ---- The "true" delay recipe (assumptions!) ----
    delay = (BASE_DELAY[t_type]
             + route_effect[route["route_id"]]
             + WEATHER_DELAY[w["weather_condition"]] * WEATHER_SENS[t_type]
             + 0.15 * w["precipitation_mm"] * WEATHER_SENS[t_type]
             + 0.05 * max(w["wind_speed_kmh"] - 30, 0)
             + (attendance / 10_000) * 2.0 * e_factor * (0.5 + ROAD_SENS[t_type])
             + (congestion - 3) * 1.2 * ROAD_SENS[t_type]
             + 2.0 * is_peak
             - 1.0 * is_holiday
             + 0.04 * route["distance_km"]
             + rng.normal(0, 3))                      # everyday randomness
    if rng.random() < 0.05:                           # rare incident (breakdown etc.)
        delay += rng.exponential(15)
    arrival_delay = int(round(max(delay, -3)))        # early by at most 3 minutes
    departure_delay = int(round(max(0.4 * arrival_delay + rng.normal(0, 2), -2)))

    # Randomly swap direction of travel
    a, b = (route["station_a"], route["station_b"]) if rng.random() < 0.5 else (route["station_b"], route["station_a"])

    rows.append({
        "trip_id": f"T{i:05d}",
        "date": date.date().isoformat(),
        "time": sched_dep.strftime("%H:%M:%S"),
        "transport_type": t_type,
        "route_id": route["route_id"],
        "origin_station": a,
        "destination_station": b,
        "distance_km": route["distance_km"],
        "scheduled_departure": sched_dep.strftime("%H:%M:%S"),
        "scheduled_arrival": sched_arr.strftime("%H:%M:%S"),
        "departure_delay_min": departure_delay,
        "arrival_delay_min": arrival_delay,
        "weather_condition": w["weather_condition"],
        "temperature_c": round(float(temp), 1),
        "humidity_pct": round(w["humidity_pct"], 1),
        "wind_speed_kmh": w["wind_speed_kmh"],
        "precipitation_mm": w["precipitation_mm"],
        "event_type": e_type,
        "event_attendance": attendance,
        "traffic_congestion_index": round(congestion, 2),
        "is_holiday": is_holiday,
        "weekday": weekday,
        "season": season_of(date.month),
        "delayed": int(arrival_delay > DELAY_THRESHOLD_MIN),
    })

df = pd.DataFrame(rows).sort_values(["date", "time"]).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 6. Make it a little "messy", like real data, so preprocessing has work to do
# ---------------------------------------------------------------------------
for col in ["temperature_c", "humidity_pct", "wind_speed_kmh"]:
    mask = rng.random(len(df)) < 0.015          # ~1.5% missing values
    df.loc[mask, col] = np.nan
df = pd.concat([df, df.sample(30, random_state=RANDOM_STATE)])  # 30 duplicate rows

RAW_DATA.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(RAW_DATA, index=False)
print(f"Saved {len(df):,} rows (SYNTHETIC) to {RAW_DATA}")
print(f"Share of delayed trips: {df['delayed'].mean():.1%}")
