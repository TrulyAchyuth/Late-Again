"""
File paths, data sources and settings shared by all the scripts.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Folders
RAW_DIR = ROOT / "data" / "raw"              # downloaded files (not committed to git)
PROCESSED_DIR = ROOT / "data" / "processed"  # cleaned tables (committed, gzip-compressed)
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

INCIDENTS_FILE = PROCESSED_DIR / "incidents.csv.gz"   # one row per delay incident
HOURLY_FILE = PROCESSED_DIR / "hourly.csv.gz"         # one row per hour of 2023-2024
METRICS_FILE = REPORTS_DIR / "metrics.json"

# Data sources (all free, no API key)
CITY = "Toronto"
LATITUDE, LONGITUDE = 43.6532, -79.3832       # downtown Toronto
TIMEZONE = "America/Toronto"
YEARS = [2023, 2024]

TTC_DATASET_PAGE = "https://open.toronto.ca/dataset/ttc-bus-delay-data/"
TTC_FILES = {   # City of Toronto Open Data (CKAN), Open Government Licence - Toronto
    2023: "https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/e271cdae-8788-4980-96ce-6a5c95bc6618/"
          "resource/10802a64-9ac0-4f2e-9538-04800a399d1e/download/ttc-bus-delay-data-2023.xlsx",
    2024: "https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/e271cdae-8788-4980-96ce-6a5c95bc6618/"
          "resource/7823b829-9952-4e4c-ac8f-0fe2ef53901c/download/ttc-bus-delay-data-2024.xlsx",
}
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"   # Open-Meteo, CC BY 4.0
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "precipitation", "rain", "snowfall",
                "snow_depth", "wind_speed_10m", "wind_gusts_10m", "weather_code"]
MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"   # Toronto Blue Jays = team 141
BLUE_JAYS_TEAM_ID = 141

# Cleaning and targets
MAX_DELAY_MIN = 120        # longer "delays" are mostly diversions/data-entry outliers (~2% of rows)
MAJOR_DELAY_MIN = 15       # an incident is a "major delay" if the next bus is >= 15 min late
RANDOM_STATE = 42

# Time-based split (no shuffling: we always predict the future from the past)
TRAIN_END = "2023-09-30"   # train:      Jan-Sep 2023
VALID_END = "2023-12-31"   # validation: Oct-Dec 2023 (used to pick the best model)
                           # test:       all of 2024 (touched once, at the very end)

# Features for the incident models (all known when an incident is reported)
INCIDENT_CATEGORICAL = ["incident", "route", "direction", "weather", "day_of_week"]
INCIDENT_NUMERIC = ["hour", "month", "is_weekend", "is_holiday", "is_rush_hour", "at_station",
                    "temperature_c", "precipitation_mm", "snowfall_cm", "snow_depth_cm",
                    "wind_kmh", "gust_kmh", "jays_game_window"]
INCIDENT_FEATURES = INCIDENT_CATEGORICAL + INCIDENT_NUMERIC

# Groups used in the ablation study (train.py) and the prediction explanation (app)
WEATHER_FEATURES = ["weather", "temperature_c", "precipitation_mm", "snowfall_cm", "snow_depth_cm",
                    "wind_kmh", "gust_kmh"]
EVENT_FEATURES = ["jays_game_window", "is_holiday"]

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEATHER_ORDER = ["Clear", "Cloudy", "Fog", "Drizzle", "Rain", "Heavy rain", "Snow", "Heavy snow", "Thunderstorm"]
