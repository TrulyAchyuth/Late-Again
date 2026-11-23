"""
build_dataset.py
----------------
Turns the raw downloads into two clean tables:

  data/processed/incidents.csv.gz  one row per TTC bus delay incident, joined with the
                                   weather of that hour, holidays and Blue Jays games
  data/processed/hourly.csv.gz     one row per hour of 2023-2024 (including quiet hours with
                                   zero incidents) with incident counts, weather and events

Run:
    python src/build_dataset.py
"""

import json

import holidays
import pandas as pd

from config import (HOURLY_FILE, INCIDENTS_FILE, MAJOR_DELAY_MIN, MAX_DELAY_MIN, PROCESSED_DIR,
                    RAW_DIR, YEARS)
from features import add_time_features, at_station, clean_direction, game_window_flags, weather_category


# ---------------------------------------------------------------------------
# Loaders: one function per source
# ---------------------------------------------------------------------------
def load_incidents() -> pd.DataFrame:
    frames = [pd.read_excel(RAW_DIR / f"ttc_bus_delays_{y}.xlsx") for y in YEARS]
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"Date": "date", "Route": "route", "Time": "time", "Location": "location",
                            "Incident": "incident", "Min Delay": "delay_min", "Min Gap": "gap_min",
                            "Direction": "direction", "Vehicle": "vehicle"})
    return df[["date", "time", "route", "location", "incident", "delay_min", "gap_min", "direction", "vehicle"]]


def load_weather() -> pd.DataFrame:
    hourly = json.loads((RAW_DIR / "weather_hourly.json").read_text())["hourly"]
    w = pd.DataFrame(hourly)
    w["hour_ts"] = pd.to_datetime(w["time"])
    w = w.drop_duplicates("hour_ts")   # daylight-saving changes can repeat an hour
    w = w.rename(columns={"temperature_2m": "temperature_c", "relative_humidity_2m": "humidity_pct",
                          "precipitation": "precipitation_mm", "rain": "rain_mm",
                          "snowfall": "snowfall_cm", "wind_speed_10m": "wind_kmh",
                          "wind_gusts_10m": "gust_kmh"})
    w["snow_depth_cm"] = w["snow_depth"] * 100          # API gives metres
    w["weather"] = w["weather_code"].map(weather_category)
    cols = ["hour_ts", "temperature_c", "humidity_pct", "precipitation_mm", "rain_mm", "snowfall_cm",
            "snow_depth_cm", "wind_kmh", "gust_kmh", "weather_code", "weather"]
    return w[cols]


def load_game_starts() -> pd.Series:
    """Local start times of Blue Jays HOME games that were actually played."""
    starts = []
    for year in YEARS:
        schedule = json.loads((RAW_DIR / f"mlb_schedule_{year}.json").read_text())
        for day in schedule["dates"]:
            for game in day["games"]:
                is_home = game["teams"]["home"]["team"]["id"] == 141
                if is_home and game["status"]["detailedState"] == "Final":
                    starts.append(game["gameDate"])
    utc = pd.to_datetime(pd.Series(starts), utc=True)
    return utc.dt.tz_convert("America/Toronto").dt.tz_localize(None)


def holiday_dates():
    """Ontario public holidays (the 'holidays' package)."""
    return list(holidays.CA(subdiv="ON", years=YEARS).keys())


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_incidents(df: pd.DataFrame, log=print) -> pd.DataFrame:
    df = df.copy()
    n0 = len(df)

    # Timestamp from the date + "HH:MM" text; drop rows we cannot read
    df["timestamp"] = pd.to_datetime(df["date"].astype(str).str[:10] + " " + df["time"].astype(str),
                                     errors="coerce")
    df = df.dropna(subset=["timestamp", "incident", "route"])
    log(f"  dropped {n0 - len(df)} rows with a missing time, incident or route")

    n1 = len(df)
    df = df.drop_duplicates(subset=["timestamp", "route", "location", "incident", "delay_min", "vehicle"])
    log(f"  dropped {n1 - len(df)} exact duplicate reports")

    n2 = len(df)
    df = df[(df["delay_min"] >= 0) & (df["delay_min"] <= MAX_DELAY_MIN)]
    log(f"  dropped {n2 - len(df)} rows with delay < 0 or > {MAX_DELAY_MIN} min (outliers)")

    df["route"] = df["route"].astype(str).str.strip().str.upper()
    df["direction"] = df["direction"].map(clean_direction)
    df["at_station"] = df["location"].map(at_station)
    df["hour_ts"] = df["timestamp"].dt.floor("h")
    df["major_delay"] = (df["delay_min"] >= MAJOR_DELAY_MIN).astype(int)
    log(f"  kept {len(df):,} of {n0:,} incidents ({len(df) / n0:.1%})")
    return df.sort_values("timestamp").reset_index(drop=True)


def build(log=print):
    log("Loading raw files")
    incidents = clean_incidents(load_incidents(), log)
    weather = load_weather()
    games = load_game_starts()
    hols = holiday_dates()
    log(f"  {len(weather):,} weather hours, {len(games)} Blue Jays home games, {len(hols)} holidays")

    # ---- incident table ----
    inc = incidents.merge(weather, on="hour_ts", how="left")
    missing_weather = inc["temperature_c"].isna().sum()
    inc = inc.dropna(subset=["temperature_c"])
    log(f"  {missing_weather} incidents had no matching weather hour and were dropped")
    inc = add_time_features(inc, "timestamp", hols)
    inc["jays_game_window"] = game_window_flags(inc["timestamp"], games)

    # ---- hourly table: every hour, even those with no incidents ----
    counts = incidents.groupby("hour_ts").agg(incidents=("delay_min", "size"),
                                              major_incidents=("major_delay", "sum"),
                                              delay_minutes=("delay_min", "sum"))
    hourly = weather.merge(counts, on="hour_ts", how="left").fillna(
        {"incidents": 0, "major_incidents": 0, "delay_minutes": 0})
    hourly = add_time_features(hourly, "hour_ts", hols)
    hourly["jays_game_window"] = game_window_flags(hourly["hour_ts"], games)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    keep = ["timestamp", "hour_ts", "route", "direction", "location", "at_station", "incident",
            "delay_min", "gap_min", "major_delay", "hour", "month", "day_of_week", "is_weekend",
            "is_holiday", "is_rush_hour", "temperature_c", "humidity_pct", "precipitation_mm",
            "rain_mm", "snowfall_cm", "snow_depth_cm", "wind_kmh", "gust_kmh", "weather_code",
            "weather", "jays_game_window"]
    inc[keep].to_csv(INCIDENTS_FILE, index=False)
    hourly.to_csv(HOURLY_FILE, index=False)
    log(f"Saved {len(inc):,} incidents -> {INCIDENTS_FILE.name}")
    log(f"Saved {len(hourly):,} hours -> {HOURLY_FILE.name}")
    return inc, hourly


if __name__ == "__main__":
    build()
