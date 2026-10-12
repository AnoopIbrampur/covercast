"""Load the Recruit Restaurant Visitor Forecasting files into dense arrays.

Kaggle's competition ships these CSVs (only the ones used here are listed):

    air_visit_data.csv   air_store_id, visit_date, visitors
    air_store_info.csv   air_store_id, air_genre_name, air_area_name, latitude, longitude
    air_reserve.csv      air_store_id, visit_datetime, reserve_datetime, reserve_visitors
    hpg_reserve.csv      hpg_store_id, visit_datetime, reserve_datetime, reserve_visitors
    store_id_relation.csv  air_store_id, hpg_store_id
    date_info.csv        calendar_date, day_of_week, holiday_flg

A store with no row for a day was closed (or didn't report). That is kept as
NaN rather than zero: zero would teach the model that Mondays are dead when
the place is simply shut on Mondays.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_FILES = (
    "air_visit_data.csv",
    "air_store_info.csv",
    "air_reserve.csv",
    "hpg_reserve.csv",
    "store_id_relation.csv",
    "date_info.csv",
)


@dataclass
class Panel:
    """Everything the features need, indexed by (store, day offset)."""

    store_ids: np.ndarray  # (S,) str
    dates: pd.DatetimeIndex  # (T,) daily, contiguous
    visitors: np.ndarray  # (S, T) float, NaN = closed / no report
    genre: np.ndarray  # (S,) int codes
    pref: np.ndarray  # (S,) int codes, prefecture-level area
    genre_names: list[str]
    pref_names: list[str]
    holiday: pd.Series  # 0/1 by calendar date, covers dates +/- 1 day
    # Reservations, one entry per booking. Day indices are offsets from dates[0].
    res_store: np.ndarray  # (R,) int
    res_visit_day: np.ndarray  # (R,) int
    res_booked_day: np.ndarray  # (R,) int
    res_visitors: np.ndarray  # (R,) float

    @property
    def n_stores(self) -> int:
        return len(self.store_ids)

    @property
    def n_days(self) -> int:
        return len(self.dates)


def check_files(data_dir: Path) -> None:
    missing = [f for f in REQUIRED_FILES if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"{data_dir} is missing {', '.join(missing)}. Download the Recruit "
            "competition data from Kaggle (see README) and unzip it there."
        )


def _parse_ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S")


def load_panel(data_dir: Path) -> Panel:
    data_dir = Path(data_dir)
    check_files(data_dir)

    visits = pd.read_csv(data_dir / "air_visit_data.csv")
    visits["visit_date"] = pd.to_datetime(visits["visit_date"], format="%Y-%m-%d")
    info = pd.read_csv(data_dir / "air_store_info.csv")
    cal = pd.read_csv(data_dir / "date_info.csv")
    cal["calendar_date"] = pd.to_datetime(cal["calendar_date"], format="%Y-%m-%d")

    store_ids = np.array(sorted(visits["air_store_id"].unique()))
    s_index = {s: i for i, s in enumerate(store_ids)}
    dates = pd.date_range(visits["visit_date"].min(), visits["visit_date"].max(), freq="D")

    V = np.full((len(store_ids), len(dates)), np.nan)
    si = visits["air_store_id"].map(s_index).to_numpy()
    ti = (visits["visit_date"] - dates[0]).dt.days.to_numpy()
    V[si, ti] = visits["visitors"].to_numpy(dtype=float)

    info = info.set_index("air_store_id").reindex(store_ids)
    genre_cat = pd.Categorical(info["air_genre_name"].fillna("unknown"))
    # "Tōkyō-to Minato-ku Shibakōen" -> "Tōkyō-to". Finer areas have too few
    # stores each to learn anything from.
    pref_cat = pd.Categorical(info["air_area_name"].fillna("unknown").str.split(" ").str[0])

    holiday = cal.set_index("calendar_date")["holiday_flg"].astype(int)
    full_range = pd.date_range(dates[0] - pd.Timedelta(days=1), dates[-1] + pd.Timedelta(days=30))
    holiday = holiday.reindex(full_range, fill_value=0)

    res = _load_reservations(data_dir, s_index, origin=dates[0])

    return Panel(
        store_ids=store_ids,
        dates=dates,
        visitors=V,
        genre=genre_cat.codes.astype(int),
        pref=pref_cat.codes.astype(int),
        genre_names=list(genre_cat.categories),
        pref_names=list(pref_cat.categories),
        holiday=holiday,
        res_store=res["store"].to_numpy(dtype=int),
        res_visit_day=res["visit_day"].to_numpy(dtype=int),
        res_booked_day=res["booked_day"].to_numpy(dtype=int),
        res_visitors=res["reserve_visitors"].to_numpy(dtype=float),
    )


def _load_reservations(data_dir: Path, s_index: dict[str, int], origin: pd.Timestamp) -> pd.DataFrame:
    """Air and HPG bookings combined, keyed by air store and day offsets from origin.

    HPG is a separate booking site; only the HPG stores that the relation file
    maps to an Air store can be used, which is a minority of them.
    """
    air = pd.read_csv(data_dir / "air_reserve.csv")
    hpg = pd.read_csv(data_dir / "hpg_reserve.csv")
    rel = pd.read_csv(data_dir / "store_id_relation.csv")
    hpg = hpg.merge(rel, on="hpg_store_id", how="inner").drop(columns="hpg_store_id")
    res = pd.concat([air, hpg[air.columns]], ignore_index=True)

    res["store"] = res["air_store_id"].map(s_index)
    res = res.dropna(subset=["store"])
    visit = _parse_ts(res["visit_datetime"]).dt.normalize()
    booked = _parse_ts(res["reserve_datetime"]).dt.normalize()
    out = pd.DataFrame(
        {
            "store": res["store"].astype(int).to_numpy(),
            "visit_day": (visit - origin).dt.days.to_numpy(),
            "booked_day": (booked - origin).dt.days.to_numpy(),
            "reserve_visitors": res["reserve_visitors"].astype(float).to_numpy(),
        }
    )
    # Bookings for days before the visit panel starts can never be a feature.
    return out[out["visit_day"] >= 0].reset_index(drop=True)
