"""Tests for the 'compared with a normal hour' logic in src/analysis.py."""

import pandas as pd

from analysis import compare_to_expected


def test_ratio_is_one_when_groups_are_identical():
    hourly = pd.DataFrame({
        "hour_ts": pd.date_range("2024-01-01", periods=8, freq="7D"),   # same weekday and hour
        "day_of_week": "Monday", "hour": 0,
        "incidents": [5, 5, 5, 5, 5, 5, 5, 5],
        "snowy": [0, 1] * 4,
    })
    t = compare_to_expected(hourly, "snowy", n_boot=50)
    assert t["ratio"].round(6).tolist() == [1.0, 1.0]


def test_ratio_detects_more_incidents():
    hourly = pd.DataFrame({
        "hour_ts": pd.date_range("2024-01-01", periods=4, freq="7D"),
        "day_of_week": "Monday", "hour": 0,
        "incidents": [4, 4, 12, 12],   # expected = 8 for every row
        "snowy": [0, 0, 1, 1],
    })
    t = compare_to_expected(hourly, "snowy", n_boot=50).set_index("snowy")
    assert t.loc[1, "ratio"] == 1.5
    assert t.loc[0, "ratio"] == 0.5
