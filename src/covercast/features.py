"""One row per (cutoff, store, horizon), built only from what was known at the cutoff.

The cutoff is the end of day `ci`: visits through `ci` are known, and so are
bookings made on or before `ci`. A booking made on day ci + 1 for day ci + 5 is
invisible to the forecast made at ci, even though it is in the file. Getting
this wrong is the easy way to make reservations look far more useful than
they are in production.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .config import ACTIVITY_WINDOW, HORIZON, MAX_LOOKBACK, MIN_OPEN_DAYS
from .data import Panel

FEATURES = [
    "h",
    "dow",
    "holiday",
    "holiday_prev",
    "holiday_next",
    "month",
    "dom",
    "genre",
    "pref",
    "store_age",
    "mean_7",
    "mean_28",
    "mean_91",
    "std_28",
    "open_frac_28",
    "sd_last",
    "sd_mean4",
    "sd_mean8",
    "sd_median8",
    "sd_open_frac13",
    "dow_effect",
    "res_known",
    "res_parties_known",
    "res_typical_at_lead",
    "res_share_28",
]
CATEGORICAL = ["dow", "genre", "pref"]
BASELINES = ["last_week", "mean4"]


def _nanmean(a: np.ndarray, axis: int = 1) -> np.ndarray:
    n = np.sum(~np.isnan(a), axis=axis)
    s = np.nansum(a, axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def _nanstd(a: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(a, axis=1)


def _nanmedian(a: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(a, axis=1)


def _window(M: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Columns lo..hi inclusive, with lo clamped at the start of the data."""
    return M[:, max(lo, 0) : hi + 1]


def _cols(M: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Gather columns, returning NaN for indices before the data starts."""
    out = np.full((M.shape[0], len(idx)), np.nan)
    ok = idx >= 0
    out[:, ok] = M[:, idx[ok]]
    return out


def first_valid_cutoff(panel: Panel) -> int:
    return MAX_LOOKBACK


def active_stores(panel: Panel, ci: int) -> np.ndarray:
    recent = _window(panel.visitors, ci - ACTIVITY_WINDOW + 1, ci)
    return np.flatnonzero(np.sum(~np.isnan(recent), axis=1) >= MIN_OPEN_DAYS)


def build_cutoff(panel: Panel, ci: int, horizon: int = HORIZON) -> pd.DataFrame:
    if ci < first_valid_cutoff(panel):
        raise ValueError(f"cutoff index {ci} leaves less than {MAX_LOOKBACK} days of history")

    stores = active_stores(panel, ci)
    V = panel.visitors[stores, : ci + 1]  # history only; nothing after ci is reachable below
    L = np.log1p(V)
    S = len(stores)
    if S == 0:
        return pd.DataFrame(columns=["cutoff", "store", "target_date", *FEATURES, *BASELINES, "actual"])

    mean_7 = _nanmean(_window(L, ci - 6, ci))
    mean_28 = _nanmean(_window(L, ci - 27, ci))
    mean_56 = _nanmean(_window(L, ci - 55, ci))
    mean_91 = _nanmean(_window(L, ci - 90, ci))
    std_28 = _nanstd(_window(L, ci - 27, ci))
    open_frac_28 = np.mean(~np.isnan(_window(V, ci - 27, ci)), axis=1)
    has_visit = ~np.isnan(V)
    first_open = np.where(has_visit.any(axis=1), has_visit.argmax(axis=1), ci)
    store_age = ci - first_open
    mean_28_or_56 = np.where(np.isnan(mean_28), mean_56, mean_28)
    raw_fallback = np.expm1(mean_28_or_56)

    res = _reservation_features(panel, stores, ci, horizon, V)

    frames = []
    for h in range(1, horizon + 1):
        ti = ci + h
        date = panel.dates[0] + pd.Timedelta(days=ti)
        # Same weekday as the target, most recent first, never after the cutoff.
        j0 = -(-h // 7)
        lag_idx = ti - 7 * np.arange(j0, j0 + 13)
        sd_V = _cols(V, lag_idx)
        sd_L = np.log1p(sd_V)

        first4 = sd_V[:, :4]
        has4 = ~np.isnan(first4)
        last_open = np.where(has4.any(axis=1), first4[np.arange(S), has4.argmax(axis=1)], np.nan)
        sd_last = np.log1p(last_open)
        sd_mean4 = _nanmean(sd_L[:, :4])
        sd_mean8 = _nanmean(sd_L[:, :8])
        sd_median8 = _nanmedian(sd_L[:, :8])
        sd_open_frac13 = np.mean(~np.isnan(sd_V), axis=1)
        dow_effect = _nanmean(sd_L) - mean_91

        actual = panel.visitors[stores, ti] if ti < panel.n_days else np.full(S, np.nan)
        hol = panel.holiday
        frames.append(
            pd.DataFrame(
                {
                    "cutoff": panel.dates[ci],
                    "store": stores,
                    "target_date": date,
                    "h": h,
                    "dow": date.dayofweek,
                    "holiday": int(hol.get(date, 0)),
                    "holiday_prev": int(hol.get(date - pd.Timedelta(days=1), 0)),
                    "holiday_next": int(hol.get(date + pd.Timedelta(days=1), 0)),
                    "month": date.month,
                    "dom": date.day,
                    "genre": panel.genre[stores],
                    "pref": panel.pref[stores],
                    "store_age": store_age,
                    "mean_7": mean_7,
                    "mean_28": mean_28,
                    "mean_91": mean_91,
                    "std_28": std_28,
                    "open_frac_28": open_frac_28,
                    "sd_last": sd_last,
                    "sd_mean4": sd_mean4,
                    "sd_mean8": sd_mean8,
                    "sd_median8": sd_median8,
                    "sd_open_frac13": sd_open_frac13,
                    "dow_effect": dow_effect,
                    "res_known": res["known"][:, h - 1],
                    "res_parties_known": res["parties"][:, h - 1],
                    "res_typical_at_lead": res["typical"][:, h - 1],
                    "res_share_28": res["share_28"],
                    # Baselines, in covers. "What we did last week" and "the
                    # average of the last four", which is how most schedules
                    # actually get written.
                    "last_week": np.where(np.isnan(last_open), raw_fallback, last_open),
                    "mean4": np.where(np.isnan(sd_mean4), raw_fallback, _nanmean(first4)),
                    "actual": actual,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _reservation_features(
    panel: Panel, stores: np.ndarray, ci: int, horizon: int, V_hist: np.ndarray
) -> dict[str, np.ndarray]:
    """Bookings known at the cutoff for each target day, plus what's normal for this store.

    `typical[:, h-1]` is the store's average reserved covers per day over the
    last 28 days, counting only bookings that existed h days before each of
    those days. That makes it directly comparable with `known[:, h-1]`: the
    model can read "this Friday already has twice its usual bookings at this
    lead time" instead of having to learn every store's booking habits.
    """
    S = len(stores)
    pos = np.full(panel.n_stores, -1)
    pos[stores] = np.arange(S)
    p = pos[panel.res_store]
    keep = (p >= 0) & (panel.res_booked_day <= ci) & (panel.res_visit_day >= ci - 27 - horizon)
    keep &= panel.res_visit_day <= ci + horizon
    p, vd, bd, rv = p[keep], panel.res_visit_day[keep], panel.res_booked_day[keep], panel.res_visitors[keep]

    known = np.zeros((S, horizon))
    parties = np.zeros((S, horizon))
    fut = vd > ci
    np.add.at(known, (p[fut], vd[fut] - ci - 1), rv[fut])
    np.add.at(parties, (p[fut], vd[fut] - ci - 1), 1.0)

    past = (vd <= ci) & (vd >= ci - 27)
    lead = vd - bd
    typical = np.zeros((S, horizon))
    for h in range(1, horizon + 1):
        m = past & (lead >= h)
        np.add.at(typical[:, h - 1], p[m], rv[m])
    typical /= 28.0

    res_28 = np.zeros(S)
    np.add.at(res_28, p[past], rv[past])
    vis_28 = np.nansum(_window(V_hist, ci - 27, ci), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        share_28 = np.where(vis_28 > 0, res_28 / vis_28, 0.0)

    return {
        "known": np.log1p(known),
        "parties": parties,
        "typical": np.log1p(typical),
        "share_28": share_28,
    }


def cutoff_indices(panel: Panel, n_eval: int, horizon: int = HORIZON, step: int = 7) -> tuple[list[int], list[int]]:
    """All weekly cutoffs, and which of them are evaluation cutoffs.

    Cutoffs are aligned so the last evaluation forecast's horizon ends on the
    last day of data. Every cutoff falls on the same weekday, as a weekly
    scheduling routine would.
    """
    last = panel.n_days - 1 - horizon
    all_ci = list(range(last, first_valid_cutoff(panel) - 1, -step))[::-1]
    if len(all_ci) <= n_eval:
        raise ValueError(f"only {len(all_ci)} weekly cutoffs available, need more than {n_eval}")
    return all_ci, all_ci[-n_eval:]
