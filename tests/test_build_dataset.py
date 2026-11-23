"""Tests for the cleaning step, using a tiny hand-made table instead of the real files."""

import pandas as pd

from build_dataset import clean_incidents
from config import MAJOR_DELAY_MIN, MAX_DELAY_MIN


def raw_rows():
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-05"] * 5),
        "time": ["08:10", "08:10", "09:00", "not a time", "10:00"],
        "route": [32, 32, "35", 36, 7],
        "location": ["KENNEDY STATION", "KENNEDY STATION", "JANE AND FINCH", "X", "Y"],
        "incident": ["Mechanical", "Mechanical", "Diversion", "Security", "Security"],
        "delay_min": [20, 20, 5, 10, MAX_DELAY_MIN + 50],
        "gap_min": [40, 40, 10, 20, 300],
        "direction": ["n", "n", "EB", "S", None],
        "vehicle": [1, 1, 2, 3, 4],
    })


def test_clean_incidents_removes_bad_rows():
    out = clean_incidents(raw_rows(), log=lambda *_: None)
    # duplicate, unreadable time and outlier are all gone
    assert len(out) == 2
    assert out["route"].tolist() == ["32", "35"]
    assert out["direction"].tolist() == ["N", "E"]
    assert out["at_station"].tolist() == [1, 0]


def test_major_delay_label():
    out = clean_incidents(raw_rows(), log=lambda *_: None)
    assert out["major_delay"].tolist() == [int(20 >= MAJOR_DELAY_MIN), int(5 >= MAJOR_DELAY_MIN)]
