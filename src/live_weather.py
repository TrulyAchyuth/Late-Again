"""
live_weather.py
---------------
Gets real weather for a city, date and hour from the free Open-Meteo API
(https://open-meteo.com, no API key needed), and turns it into the same
inputs our model uses.

Note: the MODEL was still trained on synthetic data. Live weather only makes
the inputs realistic; it does not make the prediction a real forecast.

Quick test:
    python src/live_weather.py
"""

import datetime as dt
import json
import urllib.parse
import urllib.request

# A few example cities (latitude, longitude). Add your own!
CITIES = {
    "London": (51.5074, -0.1278),
    "New York": (40.7128, -74.0060),
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Berlin": (52.5200, 13.4050),
    "Tokyo": (35.6762, 139.6503),
    "Sydney": (-33.8688, 151.2093),
}

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"   # for older dates
HOURLY_VARS = "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code"


def weather_code_to_condition(code: int) -> str:
    """Convert a WMO weather code (used by Open-Meteo) to one of our 7 conditions."""
    if code in (0, 1):
        return "Clear"
    if code in (2, 3):
        return "Cloudy"
    if code in (45, 48):
        return "Fog"
    if code in (65, 82):
        return "Heavy Rain"
    if 51 <= code <= 67 or code in (80, 81):
        return "Rain"                      # drizzle, rain, freezing rain, showers
    if 71 <= code <= 77 or code in (85, 86):
        return "Snow"
    if code >= 95:
        return "Storm"                     # thunderstorms
    return "Cloudy"


def fetch_json(lat: float, lon: float, day: dt.date) -> dict:
    """Download one day of hourly weather. Uses the archive API for dates > 80 days ago."""
    use_archive = (dt.date.today() - day).days > 80
    params = {
        "latitude": lat, "longitude": lon, "hourly": HOURLY_VARS,
        "daily": "precipitation_sum",
        "start_date": day.isoformat(), "end_date": day.isoformat(), "timezone": "auto",
    }
    url = (ARCHIVE_URL if use_archive else FORECAST_URL) + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_hour(data: dict, hour: int) -> dict:
    """Pick one hour out of the API response and return model-ready inputs."""
    hourly = data["hourly"]
    i = min(hour, len(hourly["time"]) - 1)
    code = hourly["weather_code"][i]
    if code is None:
        raise ValueError("No weather data for that hour (date may be too far in the future).")
    return {
        "weather_condition": weather_code_to_condition(int(code)),
        "temperature_c": float(hourly["temperature_2m"][i]),
        "humidity_pct": float(hourly["relative_humidity_2m"][i]),
        "wind_speed_kmh": float(hourly["wind_speed_10m"][i]),
        # Our dataset stores rain as a DAILY total, so use the day's total here too
        "precipitation_mm": float(data["daily"]["precipitation_sum"][0] or 0.0),
        "rain_this_hour_mm": float(hourly["precipitation"][i] or 0.0),
        "time": hourly["time"][i],
    }


def get_weather(lat: float, lon: float, day: dt.date, hour: int) -> dict:
    """Fetch + parse. Raises an exception if the internet/API is not reachable."""
    if (day - dt.date.today()).days > 15:
        raise ValueError("Open-Meteo forecasts only go about 16 days ahead.")
    return parse_hour(fetch_json(lat, lon, day), hour)


if __name__ == "__main__":
    print(get_weather(*CITIES["London"], dt.date.today(), 8))
