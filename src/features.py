"""
Small helper functions that turn raw values into model features.
"""
import numpy as np
import pandas as pd

RUSH_HOURS = (7, 8, 9, 16, 17, 18)   # weekday morning and evening peaks


def weather_category(code) -> str:
    """Map a WMO weather code (used by Open-Meteo) to a readable category."""
    if code is None or pd.isna(code):
        return "Cloudy"
    code = int(code)
    if code in (0, 1):
        return "Clear"
    if code in (2, 3):
        return "Cloudy"
    if code in (45, 48):
        return "Fog"
    if 51 <= code <= 57:
        return "Drizzle"
    if code in (61, 63, 66, 80, 81):
        return "Rain"
    if code in (65, 67, 82):
        return "Heavy rain"
    if code in (71, 73, 77, 85):
        return "Snow"
    if code in (75, 86):
        return "Heavy snow"
    if code >= 95:
        return "Thunderstorm"
    return "Cloudy"


def clean_direction(value) -> str:
    """TTC directions are typed by hand ('N', 'n', 'NB', 'b', '`'...). Keep N/S/E/W/B, else Unknown."""
    if value is None or pd.isna(value):
        return "Unknown"
    first = str(value).strip().upper()[:1]
    return first if first in {"N", "S", "E", "W", "B"} else "Unknown"


def at_station(location) -> int:
    """1 if the incident happened at a subway/bus station (stations are hubs with many buses)."""
    if location is None or pd.isna(location):
        return 0
    text = str(location).upper()
    return int("STATION" in text or text.endswith(" STN") or " STN " in text)


def is_rush_hour(hour, is_weekend, is_holiday) -> int:
    return int(hour in RUSH_HOURS and not is_weekend and not is_holiday)


def add_time_features(df: pd.DataFrame, ts_col: str, holiday_dates) -> pd.DataFrame:
    """Add hour, month, day_of_week, is_weekend, is_holiday and is_rush_hour from a timestamp column."""
    df = df.copy()
    ts = pd.to_datetime(df[ts_col])
    holiday_dates = {pd.Timestamp(d).date() for d in holiday_dates}
    df["hour"] = ts.dt.hour
    df["month"] = ts.dt.month
    df["day_of_week"] = ts.dt.day_name()
    df["is_weekend"] = (ts.dt.day_of_week >= 5).astype(int)
    df["is_holiday"] = ts.dt.date.isin(holiday_dates).astype(int)
    df["is_rush_hour"] = [is_rush_hour(h, w, hol) for h, w, hol in
                          zip(df["hour"], df["is_weekend"], df["is_holiday"])]
    return df


def game_window_flags(timestamps, game_starts, before_h=1, after_h=4) -> np.ndarray:
    """
    1 if a timestamp falls between 1 hour before and 4 hours after the start of a
    Blue Jays home game (people travelling to and from the stadium), else 0.
    Both inputs must be timezone-naive local times.
    """
    ts = pd.to_datetime(pd.Series(timestamps)).to_numpy()
    flags = np.zeros(len(ts), dtype=int)
    for start in pd.to_datetime(pd.Series(game_starts)):
        lo = np.datetime64(start - pd.Timedelta(hours=before_h))
        hi = np.datetime64(start + pd.Timedelta(hours=after_h))
        flags |= ((ts >= lo) & (ts <= hi)).astype(int)
    return flags
