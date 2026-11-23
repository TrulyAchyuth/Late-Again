"""
live_weather.py
---------------
Fetches the weather for downtown Toronto at a given date and hour from Open-Meteo
(free, no API key) and returns it in the same units the model was trained on.

    * dates more than ~5 days ago  -> historical archive API
    * recent / next 16 days        -> forecast API

Quick test:
    python src/live_weather.py
"""

import datetime as dt
import json
import urllib.parse
import urllib.request

from config import (LATITUDE, LONGITUDE, TIMEZONE, WEATHER_ARCHIVE_URL, WEATHER_FORECAST_URL,
                    WEATHER_VARS)
from features import weather_category


def fetch_json(day: dt.date) -> dict:
    days_ago = (dt.date.today() - day).days
    if days_ago < -15:
        raise ValueError("Open-Meteo forecasts only reach about 16 days ahead.")
    url = WEATHER_ARCHIVE_URL if days_ago > 5 else WEATHER_FORECAST_URL
    params = {"latitude": LATITUDE, "longitude": LONGITUDE, "timezone": TIMEZONE,
              "hourly": ",".join(WEATHER_VARS),
              "start_date": day.isoformat(), "end_date": day.isoformat()}
    with urllib.request.urlopen(url + "?" + urllib.parse.urlencode(params), timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_hour(data: dict, hour: int) -> dict:
    """Pick one hour from an Open-Meteo response and convert it to model inputs."""
    h = data["hourly"]
    i = min(hour, len(h["time"]) - 1)
    if h["weather_code"][i] is None:
        raise ValueError("Open-Meteo has no data for that hour yet.")

    def val(key, default=0.0):
        v = h.get(key, [None] * (i + 1))[i]
        return float(v) if v is not None else default

    return {
        "weather": weather_category(h["weather_code"][i]),
        "temperature_c": val("temperature_2m"),
        "precipitation_mm": val("precipitation"),
        "snowfall_cm": val("snowfall"),
        "snow_depth_cm": val("snow_depth") * 100,   # metres -> cm
        "wind_kmh": val("wind_speed_10m"),
        "gust_kmh": val("wind_gusts_10m"),
        "time": h["time"][i],
    }


def get_weather(day: dt.date, hour: int) -> dict:
    return parse_hour(fetch_json(day), hour)


if __name__ == "__main__":
    print(get_weather(dt.date.today(), 8))
