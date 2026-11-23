"""
Summary tables for the dashboard and the notebook.

For weather and events I compare each hour with what is normal for that year,
weekday and hour. A ratio of 1.25 means 25% more incidents than a normal hour
at the same time of the week.
"""
import numpy as np
import pandas as pd


def add_expected(hourly: pd.DataFrame) -> pd.DataFrame:
    """Add 'expected' = mean incidents for the same year, weekday and hour."""
    h = hourly.copy()
    h["year"] = pd.to_datetime(h["hour_ts"]).dt.year
    h["expected"] = h.groupby(["year", "day_of_week", "hour"])["incidents"].transform("mean")
    return h


def compare_to_expected(hourly: pd.DataFrame, group_col: str, n_boot: int = 300, seed: int = 0) -> pd.DataFrame:
    """
    For each value of group_col: number of hours, actual vs expected incidents, the ratio,
    and a 90% bootstrap interval for the ratio (wide interval = not enough hours to be sure).
    """
    h = add_expected(hourly)
    rng = np.random.default_rng(seed)
    rows = []
    for value, g in h.groupby(group_col, observed=True):
        actual, expected = g["incidents"].to_numpy(), g["expected"].to_numpy()
        ratio = actual.sum() / expected.sum()
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(g), len(g))
            boots.append(actual[idx].sum() / expected[idx].sum())
        lo, hi = np.percentile(boots, [5, 95])
        rows.append({group_col: value, "hours": len(g), "incidents_per_hour": actual.mean(),
                     "expected_per_hour": expected.mean(), "ratio": ratio, "ratio_low": lo, "ratio_high": hi})
    return pd.DataFrame(rows)


def snow_and_rain_bins(hourly: pd.DataFrame) -> pd.DataFrame:
    """Readable intensity bins for snowfall (cm/h) and precipitation (mm/h)."""
    h = hourly.copy()
    h["snowfall"] = pd.cut(h["snowfall_cm"], [-0.01, 0.0, 0.5, 1.5, 100],
                           labels=["None", "Light (<0.5 cm/h)", "Moderate (0.5-1.5)", "Heavy (>1.5)"])
    h["rainfall"] = pd.cut(h["rain_mm"], [-0.01, 0.0, 1.0, 4.0, 200],
                           labels=["None", "Light (<1 mm/h)", "Moderate (1-4)", "Heavy (>4)"])
    return h


def kpis(inc: pd.DataFrame) -> dict:
    return {
        "incidents": len(inc),
        "routes": inc["route"].nunique(),
        "median_delay": float(inc["delay_min"].median()),
        "major_share": float(inc["major_delay"].mean()),
        "first": pd.to_datetime(inc["timestamp"]).min(),
        "last": pd.to_datetime(inc["timestamp"]).max(),
    }


def route_table(inc: pd.DataFrame, min_incidents: int = 200) -> pd.DataFrame:
    t = (inc.groupby("route")
            .agg(incidents=("delay_min", "size"), median_delay=("delay_min", "median"),
                 major_share=("major_delay", "mean"))
            .query("incidents >= @min_incidents")
            .sort_values("incidents", ascending=False))
    return t.reset_index()


def incident_table(inc: pd.DataFrame) -> pd.DataFrame:
    return (inc.groupby("incident")
               .agg(incidents=("delay_min", "size"), median_delay=("delay_min", "median"),
                    major_share=("major_delay", "mean"))
               .sort_values("incidents", ascending=False).reset_index())


def hour_weekday_table(hourly: pd.DataFrame, value: str = "incidents") -> pd.DataFrame:
    return hourly.pivot_table(index="day_of_week", columns="hour", values=value, aggfunc="mean")
