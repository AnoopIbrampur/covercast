"""Daily weather at each restaurant's location, from the Open-Meteo historical archive.

Open-Meteo (https://open-meteo.com) serves reanalysis-based historical weather
by latitude and longitude, free and without a key. This is *observed*
weather. Weather forecasts as they stood in 2016-17 aren't publicly archived,
so any feature built from this for a future day is an idealised "perfect
weather forecast". See scripts/weather_experiment.py for how that is handled.

    .venv/bin/python -m covercast.weather   # fetch into artifacts/raw/weather.csv
"""

from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW_DIR

API = "https://archive-api.open-meteo.com/v1/archive"
DAILY = ["precipitation_sum", "snowfall_sum", "temperature_2m_max", "temperature_2m_min", "precipitation_hours"]
EVENING = range(17, 22)  # local hours 17:00-21:59, the dinner service
START, END = "2016-01-01", "2017-05-31"


def _ssl_context() -> ssl.SSLContext:
    # python.org builds on macOS ship without root certificates; certifi's bundle
    # keeps verification on instead of switching it off.
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _get(lats: list[float], lons: list[float]) -> list[dict]:
    q = {
        "latitude": ",".join(f"{v:.4f}" for v in lats),
        "longitude": ",".join(f"{v:.4f}" for v in lons),
        "start_date": START, "end_date": END,
        "daily": ",".join(DAILY), "hourly": "precipitation", "timezone": "Asia/Tokyo",
    }
    url = f"{API}?{urllib.parse.urlencode(q)}"
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=120, context=_ssl_context()) as r:
                out = json.loads(r.read())
            return out if isinstance(out, list) else [out]
        except urllib.error.HTTPError as e:
            # 429 is Open-Meteo's fair-use limit: back off for real, not for a few seconds.
            if attempt == 5:
                raise
            time.sleep(65 if e.code == 429 else 5 * (attempt + 1))
        except urllib.error.URLError:
            if attempt == 5:
                raise
            time.sleep(5 * (attempt + 1))
    return []


def grid_key(lat: pd.Series, lon: pd.Series) -> list[tuple[float, float]]:
    """Locations rounded to 0.1 degree, about the archive's ~10 km resolution."""
    return list(zip(lat.round(1), lon.round(1)))


def fetch(raw_dir: Path = RAW_DIR, batch: int = 4) -> Path:
    info = pd.read_csv(Path(raw_dir) / "air_store_info.csv")
    locs = pd.DataFrame(sorted(set(grid_key(info["latitude"], info["longitude"]))), columns=["latitude", "longitude"])
    parts = Path(raw_dir) / "weather_parts"
    parts.mkdir(exist_ok=True)
    for i in range(0, len(locs), batch):
        part_file = parts / f"part_{i:03d}.csv"
        if part_file.exists():
            continue
        part = locs.iloc[i : i + batch]
        frames = []
        for (lat, lon), res in zip(part.itertuples(index=False), _get(part.latitude.tolist(), part.longitude.tolist())):
            d = pd.DataFrame(res["daily"]).rename(columns={"time": "date"})
            h = pd.DataFrame(res["hourly"])
            h["time"] = pd.to_datetime(h["time"])
            ev = h[h["time"].dt.hour.isin(EVENING)].groupby(h["time"].dt.strftime("%Y-%m-%d"))["precipitation"].sum()
            d["evening_precip"] = d["date"].map(ev)
            d["latitude"], d["longitude"] = lat, lon
            frames.append(d)
        pd.concat(frames, ignore_index=True).to_csv(part_file, index=False)
        print(f"fetched {min(i + batch, len(locs))}/{len(locs)} locations", flush=True)
        time.sleep(8)
    out = Path(raw_dir) / "weather.csv"
    pd.concat([pd.read_csv(f) for f in sorted(parts.glob("part_*.csv"))], ignore_index=True).to_csv(out, index=False)
    return out


def store_weather(raw_dir: Path, store_ids: np.ndarray, dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """(S, T) arrays aligned to the panel: precip, evening_precip, snow, tmax, temp_anom.

    temp_anom is the day's max temperature minus that location's average max for
    the same calendar month across the data, so it reads "unusually warm or
    cold for the season" rather than restating the season.
    """
    w = pd.read_csv(Path(raw_dir) / "weather.csv", parse_dates=["date"])
    info = pd.read_csv(Path(raw_dir) / "air_store_info.csv").set_index("air_store_id").reindex(store_ids)
    w["month"] = w["date"].dt.month
    w["temp_anom"] = w["temperature_2m_max"] - w.groupby(["latitude", "longitude", "month"])["temperature_2m_max"].transform("mean")
    key = grid_key(info["latitude"], info["longitude"])
    w["key"] = grid_key(w["latitude"], w["longitude"])
    cols = {"precip": "precipitation_sum", "evening_precip": "evening_precip", "snow": "snowfall_sum",
            "tmax": "temperature_2m_max", "temp_anom": "temp_anom"}
    out = {c: np.full((len(store_ids), len(dates)), np.nan) for c in cols}
    by_key = {k: g.set_index("date").reindex(dates) for k, g in w.groupby("key")}
    for i, k in enumerate(key):
        g = by_key.get(k)
        if g is None:
            continue
        for c, src in cols.items():
            out[c][i] = g[src].to_numpy(dtype=float)
    return out


if __name__ == "__main__":
    print(f"wrote {fetch()}")
