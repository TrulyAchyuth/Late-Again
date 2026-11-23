"""
download_data.py
----------------
Downloads the three real data sources into data/raw/ (about 8 MB in total):

  1. TTC bus delay incidents, 2023 and 2024 (City of Toronto Open Data)
  2. Hourly weather for downtown Toronto (Open-Meteo historical archive)
  3. Toronto Blue Jays home games at Rogers Centre (MLB Stats API)

Run:
    python src/download_data.py
"""

import json
import urllib.parse
import urllib.request

from config import (BLUE_JAYS_TEAM_ID, LATITUDE, LONGITUDE, MLB_SCHEDULE_URL, RAW_DIR,
                    TIMEZONE, TTC_FILES, WEATHER_ARCHIVE_URL, WEATHER_VARS, YEARS)


def download(url: str, path):
    """Save a URL to a file (skips files that are already downloaded)."""
    if path.exists():
        print(f"  already have {path.name}")
        return
    print(f"  downloading {path.name} ...")
    with urllib.request.urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("1/3 TTC bus delay data")
    for year, url in TTC_FILES.items():
        download(url, RAW_DIR / f"ttc_bus_delays_{year}.xlsx")

    print("2/3 Hourly weather (Open-Meteo)")
    params = {
        "latitude": LATITUDE, "longitude": LONGITUDE, "timezone": TIMEZONE,
        "start_date": f"{min(YEARS)}-01-01", "end_date": f"{max(YEARS)}-12-31",
        "hourly": ",".join(WEATHER_VARS),
    }
    download(WEATHER_ARCHIVE_URL + "?" + urllib.parse.urlencode(params), RAW_DIR / "weather_hourly.json")

    print("3/3 Blue Jays home games (MLB Stats API)")
    for year in YEARS:
        params = {"sportId": 1, "teamId": BLUE_JAYS_TEAM_ID, "season": year, "gameType": "R"}
        download(MLB_SCHEDULE_URL + "?" + urllib.parse.urlencode(params), RAW_DIR / f"mlb_schedule_{year}.json")

    # quick sanity check
    weather = json.loads((RAW_DIR / "weather_hourly.json").read_text())
    print(f"Done. Weather hours: {len(weather['hourly']['time']):,}")


if __name__ == "__main__":
    main()
