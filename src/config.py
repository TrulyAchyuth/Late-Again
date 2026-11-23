"""
config.py
---------
One place for file paths and settings, so every script uses the same values.
"""

from pathlib import Path

# Project root = the folder that contains this "src" folder
ROOT = Path(__file__).resolve().parent.parent

# ---- Data files ----
RAW_DATA = ROOT / "data" / "raw" / "public_transport_delays.csv"
CLEAN_DATA = ROOT / "data" / "processed" / "clean_data.csv"

# ---- Outputs ----
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ---- Settings ----
RANDOM_STATE = 42          # makes results reproducible
TEST_SIZE = 0.2            # 20% of rows are held back for testing
DELAY_THRESHOLD_MIN = 5    # a trip counts as "delayed" if it arrives > 5 min late

# ---- Target columns ----
TARGET_CLASS = "delayed"              # 1 = delayed, 0 = on time
TARGET_REG = "arrival_delay_min"      # delay in minutes

# ---- Feature columns used by the models ----
# These are things you could know BEFORE the trip starts
# (timetable, weather forecast, planned events, expected traffic).
CATEGORICAL_FEATURES = [
    "transport_type",
    "route_id",
    "weather_condition",
    "event_type",
    "season",
    "day_of_week",
]

NUMERIC_FEATURES = [
    "hour",
    "month",
    "is_weekend",
    "is_holiday",
    "is_peak_hour",
    "distance_km",
    "scheduled_duration_min",
    "temperature_c",
    "humidity_pct",
    "wind_speed_kmh",
    "precipitation_mm",
    "event_attendance",
    "traffic_congestion_index",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Order used for plots so days and weather read naturally
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEATHER_ORDER = ["Clear", "Cloudy", "Fog", "Rain", "Heavy Rain", "Snow", "Storm"]
EVENT_ORDER = ["No Event", "Concert", "Sports", "Festival", "Parade", "Protest"]
