"""Unit tests for the small feature helpers in src/features.py."""

import pandas as pd

from features import (add_time_features, at_station, clean_direction, game_window_flags,
                      is_rush_hour, weather_category)


def test_weather_category_maps_wmo_codes():
    assert weather_category(0) == "Clear"
    assert weather_category(3) == "Cloudy"
    assert weather_category(45) == "Fog"
    assert weather_category(53) == "Drizzle"
    assert weather_category(63) == "Rain"
    assert weather_category(65) == "Heavy rain"
    assert weather_category(73) == "Snow"
    assert weather_category(75) == "Heavy snow"
    assert weather_category(95) == "Thunderstorm"
    assert weather_category(None) == "Cloudy"   # missing value -> safe default


def test_clean_direction_handles_messy_input():
    assert clean_direction("n") == "N"
    assert clean_direction("NB") == "N"
    assert clean_direction(" w ") == "W"
    assert clean_direction("`") == "Unknown"
    assert clean_direction(None) == "Unknown"


def test_at_station():
    assert at_station("KENNEDY STATION") == 1
    assert at_station("Finch Stn") == 1
    assert at_station("YONGE AND EGLINTON") == 0
    assert at_station(float("nan")) == 0


def test_rush_hour_only_on_working_days():
    assert is_rush_hour(8, is_weekend=0, is_holiday=0) == 1
    assert is_rush_hour(8, is_weekend=1, is_holiday=0) == 0
    assert is_rush_hour(8, is_weekend=0, is_holiday=1) == 0
    assert is_rush_hour(13, is_weekend=0, is_holiday=0) == 0


def test_add_time_features():
    df = pd.DataFrame({"ts": ["2024-07-01 08:15", "2024-07-02 08:15", "2024-07-06 17:00"]})
    out = add_time_features(df, "ts", holiday_dates=["2024-07-01"])   # Canada Day
    assert out["day_of_week"].tolist() == ["Monday", "Tuesday", "Saturday"]
    assert out["is_holiday"].tolist() == [1, 0, 0]
    assert out["is_weekend"].tolist() == [0, 0, 1]
    assert out["is_rush_hour"].tolist() == [0, 1, 0]


def test_game_window_flags():
    games = [pd.Timestamp("2024-06-01 19:07")]
    times = ["2024-06-01 17:30", "2024-06-01 18:30", "2024-06-01 22:30", "2024-06-01 23:30"]
    assert game_window_flags(times, games).tolist() == [0, 1, 1, 0]
