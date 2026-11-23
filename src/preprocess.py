"""
preprocess.py
-------------
Cleans the raw CSV and adds time-based features.

Steps:
    1. Remove duplicate trips.
    2. Fill missing weather values with the median for that weather condition.
    3. Create time features: hour, month, day_of_week, is_weekend, is_peak_hour.
    4. Create scheduled_duration_min from the timetable.
    5. Recompute the "delayed" label from the delay in minutes (keeps them consistent).

Run:
    python src/preprocess.py
"""

import pandas as pd

from config import RAW_DATA, CLEAN_DATA, DELAY_THRESHOLD_MIN


def load_raw(path=RAW_DATA) -> pd.DataFrame:
    """Read the raw CSV file."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw data with extra features."""
    df = df.copy()

    # 1. Duplicates: the same trip_id should appear only once
    before = len(df)
    df = df.drop_duplicates(subset="trip_id").reset_index(drop=True)
    print(f"Removed {before - len(df)} duplicate rows")

    # 2. Missing values: use the median of similar weather days
    for col in ["temperature_c", "humidity_pct", "wind_speed_kmh"]:
        n_missing = df[col].isna().sum()
        df[col] = df[col].fillna(df.groupby("weather_condition")[col].transform("median"))
        df[col] = df[col].fillna(df[col].median())  # safety net
        print(f"Filled {n_missing} missing values in {col}")

    df["event_type"] = df["event_type"].fillna("No Event")
    df["event_attendance"] = df["event_attendance"].fillna(0)

    # 3. Time features
    dt = pd.to_datetime(df["date"] + " " + df["scheduled_departure"])
    df["hour"] = dt.dt.hour
    df["month"] = dt.dt.month
    df["day_of_week"] = dt.dt.day_name()
    df["is_weekend"] = (dt.dt.day_of_week >= 5).astype(int)
    # Peak hour = weekday rush hours (7-9 am, 5-7 pm), not on holidays
    rush = df["hour"].isin([7, 8, 9, 17, 18, 19])
    df["is_peak_hour"] = (rush & (df["is_weekend"] == 0) & (df["is_holiday"] == 0)).astype(int)

    # 4. Scheduled trip length (handles trips that pass midnight)
    dep = pd.to_datetime(df["scheduled_departure"], format="%H:%M:%S")
    arr = pd.to_datetime(df["scheduled_arrival"], format="%H:%M:%S")
    df["scheduled_duration_min"] = ((arr - dep).dt.total_seconds() / 60) % (24 * 60)

    # 5. Classification label
    df["delayed"] = (df["arrival_delay_min"] > DELAY_THRESHOLD_MIN).astype(int)

    return df


def main():
    df = clean(load_raw())
    CLEAN_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_DATA, index=False)
    print(f"Saved {len(df):,} clean rows to {CLEAN_DATA}")


if __name__ == "__main__":
    main()
