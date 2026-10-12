"""Write a fake restaurant group in the exact Recruit CSV schema.

This exists so the pipeline and its tests run without the Kaggle download. The
generator has the structure that matters for the real problem (weekly cycles,
fixed closing days, holidays, persistent shocks that bookings partly reveal)
but none of its numbers say anything about real restaurants.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

START = "2016-01-01"
END = "2017-04-22"
CAL_END = "2017-05-31"

# Japanese public holidays over the Recruit period, as date_info.csv flags them.
HOLIDAYS = [
    "2016-01-01", "2016-01-02", "2016-01-03", "2016-01-11", "2016-02-11", "2016-03-21",
    "2016-04-29", "2016-05-03", "2016-05-04", "2016-05-05", "2016-07-18", "2016-08-11",
    "2016-09-19", "2016-09-22", "2016-10-10", "2016-11-03", "2016-11-23", "2016-12-23",
    "2016-12-31", "2017-01-01", "2017-01-02", "2017-01-03", "2017-01-09", "2017-02-11",
    "2017-03-20", "2017-04-29", "2017-05-03", "2017-05-04", "2017-05-05",
]

GENRES = {
    # Mon..Sun multipliers on log-scale-ish demand.
    "Izakaya": [0.8, 0.85, 0.9, 1.0, 1.45, 1.6, 0.9],
    "Cafe/Sweets": [0.85, 0.85, 0.85, 0.9, 1.0, 1.5, 1.45],
    "Italian/French": [0.7, 0.8, 0.85, 0.95, 1.3, 1.55, 1.2],
    "Dining bar": [0.75, 0.85, 0.9, 1.0, 1.5, 1.5, 0.7],
}
PREFS = ["Tōkyō-to", "Ōsaka-fu", "Fukuoka-ken", "Hokkaidō"]


def generate(out_dir: Path, n_stores: int = 120, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(START, END, freq="D")
    T = len(dates)
    dow = dates.dayofweek.to_numpy()
    hol = dates.isin(pd.to_datetime(HOLIDAYS))

    store_ids = [f"air_{i:08x}" for i in rng.choice(16**8, n_stores, replace=False)]
    genres = rng.choice(list(GENRES), n_stores)
    prefs = rng.choice(PREFS, n_stores, p=[0.55, 0.2, 0.15, 0.1])

    visit_rows, info_rows, air_res, hpg_res, relation = [], [], [], [], []
    for s, sid in enumerate(store_ids):
        genre = genres[s]
        base = np.exp(rng.normal(np.log(20), 0.5))
        dow_mult = np.array(GENRES[genre]) * np.exp(rng.normal(0, 0.08, 7))
        # Business-district places lose trade on holidays; leisure ones gain it.
        hol_mult = rng.choice([0.75, 1.35], p=[0.35, 0.65])
        closed_dow = rng.choice([-1, 0, 1, 6], p=[0.55, 0.2, 0.15, 0.1])
        open_from = 0 if rng.random() < 0.6 else rng.integers(150, 220)  # July 2016 onboarding wave
        trend = rng.normal(0, 0.15) / T
        book_frac = rng.choice([0.0, rng.uniform(0.05, 0.5)], p=[0.55, 0.45])
        via_hpg = rng.random() < 0.2

        # AR(1) on log demand: a good month stays good for a while, and
        # bookings are an early read on it.
        ar = np.zeros(T)
        eps = rng.normal(0, 0.12, T)
        for t in range(1, T):
            ar[t] = 0.9 * ar[t - 1] + eps[t]

        lam = base * dow_mult[dow] * np.where(hol, hol_mult, 1.0) * np.exp(ar + trend * np.arange(T))
        visitors = np.maximum(rng.poisson(lam), 1)
        open_ = (np.arange(T) >= open_from) & (dow != closed_dow) & (rng.random(T) > 0.03)

        for t in np.flatnonzero(open_):
            visit_rows.append((sid, dates[t].strftime("%Y-%m-%d"), int(visitors[t])))
            if book_frac == 0:
                continue
            reserved = rng.binomial(visitors[t], book_frac)
            while reserved > 0:
                party = min(reserved, int(rng.integers(1, 5)))
                reserved -= party
                visit_ts = dates[t] + pd.Timedelta(hours=int(rng.integers(17, 22)))
                lead_days = int(rng.geometric(0.22)) - 1
                booked_ts = visit_ts - pd.Timedelta(days=lead_days, hours=int(rng.integers(1, 9)))
                row = (visit_ts.strftime("%Y-%m-%d %H:%M:%S"), booked_ts.strftime("%Y-%m-%d %H:%M:%S"), party)
                if via_hpg and rng.random() < 0.5:
                    hpg_res.append((f"hpg_{s:08x}",) + row)
                else:
                    air_res.append((sid,) + row)

        if via_hpg:
            relation.append((sid, f"hpg_{s:08x}"))
        info_rows.append((sid, genre, f"{prefs[s]} Synthetic-ku Area{s % 7}", 35.0, 139.0))

    pd.DataFrame(visit_rows, columns=["air_store_id", "visit_date", "visitors"]).to_csv(
        out_dir / "air_visit_data.csv", index=False
    )
    pd.DataFrame(
        info_rows, columns=["air_store_id", "air_genre_name", "air_area_name", "latitude", "longitude"]
    ).to_csv(out_dir / "air_store_info.csv", index=False)
    res_cols = ["visit_datetime", "reserve_datetime", "reserve_visitors"]
    pd.DataFrame(air_res, columns=["air_store_id"] + res_cols).to_csv(out_dir / "air_reserve.csv", index=False)
    pd.DataFrame(hpg_res, columns=["hpg_store_id"] + res_cols).to_csv(out_dir / "hpg_reserve.csv", index=False)
    pd.DataFrame(relation, columns=["air_store_id", "hpg_store_id"]).to_csv(
        out_dir / "store_id_relation.csv", index=False
    )

    cal = pd.date_range(START, CAL_END, freq="D")
    pd.DataFrame(
        {
            "calendar_date": cal.strftime("%Y-%m-%d"),
            "day_of_week": cal.day_name(),
            "holiday_flg": cal.isin(pd.to_datetime(HOLIDAYS)).astype(int),
        }
    ).to_csv(out_dir / "date_info.csv", index=False)
    return out_dir
