"""Two classical forecasters, as benchmarks the gradient-boosted model has to beat.

**Weekday exponential smoothing.** Per store, keep an exponentially smoothed
log level for each weekday, updated only on days the store was open, plus a
smoothed "recent deviation" from those weekday levels (a store running hot or
cold this month). Forecast for a day h ahead:

    log1p(covers) = S[store, weekday] + phi**h * dev[store]

This is Holt-Winters in spirit (level per season, damped short-run
correction) but robust to closed days, which break textbook ETS. Its three
parameters are tuned on cutoffs before the test window.

**Booking pickup.** The hotel and airline approach to reservations: on top of
what's already booked, add the covers that typically arrive after that point.
For a target day h days out,

    covers = booked_now + mean over recent same-weekday days of (actual - booked_by_h_days_before)

The pickup term covers both later bookings and walk-ins. It only applies to
stores that take bookings.
"""

from __future__ import annotations

import numpy as np

from .config import HORIZON
from .data import Panel


def weekday_smoothing(panel: Panel, ci: int, stores: np.ndarray, alpha: float, alpha_dev: float, phi: float,
                      horizon: int = HORIZON) -> np.ndarray:
    """Forecast covers (len(stores), horizon) from data up to and including day ci."""
    L = np.log1p(panel.visitors[stores, : ci + 1])
    S_n = len(stores)
    dow0 = panel.dates[0].dayofweek
    level = np.full((S_n, 7), np.nan)
    dev = np.zeros(S_n)
    for t in range(ci + 1):
        x = L[:, t]
        ok = ~np.isnan(x)
        if not ok.any():
            continue
        d = (dow0 + t) % 7
        cur = level[:, d]
        fresh = ok & np.isnan(cur)
        upd = ok & ~fresh
        resid = np.where(upd, x - cur, 0.0)
        dev = np.where(upd, dev + alpha_dev * (resid - dev), dev)
        level[fresh, d] = x[fresh]
        level[upd, d] = cur[upd] + alpha * (x[upd] - cur[upd])
    overall = np.nanmean(np.where(np.isnan(level), np.nan, level), axis=1)
    out = np.empty((S_n, horizon))
    for h in range(1, horizon + 1):
        d = (dow0 + ci + h) % 7
        base = np.where(np.isnan(level[:, d]), overall, level[:, d])
        out[:, h - 1] = np.expm1(base + phi**h * dev)
    return np.clip(out, 0, None)


def booked_by_lead(panel: Panel, max_lead: int = HORIZON + 1) -> np.ndarray:
    """K[s, day, h] = covers booked for that day at least h days before it (h = 0..max_lead-1)."""
    K = np.zeros((panel.n_stores, panel.n_days, max_lead))
    lead = np.clip(panel.res_visit_day - panel.res_booked_day, 0, max_lead - 1)
    ok = panel.res_visit_day < panel.n_days
    np.add.at(K, (panel.res_store[ok], panel.res_visit_day[ok], lead[ok]), panel.res_visitors[ok])
    # Booked at least h days ahead = sum over leads >= h.
    return np.flip(np.cumsum(np.flip(K, axis=2), axis=2), axis=2)


def pickup(panel: Panel, K: np.ndarray, ci: int, stores: np.ndarray, booked_now: np.ndarray,
           horizon: int = HORIZON, weeks: int = 8) -> np.ndarray:
    """Pickup forecast (len(stores), horizon). booked_now[:, h-1] = covers booked by the cutoff for day ci+h."""
    out = np.full((len(stores), horizon), np.nan)
    for h in range(1, horizon + 1):
        ti = ci + h
        j0 = -(-h // 7)
        past = ti - 7 * np.arange(j0, j0 + weeks)
        past = past[past >= 0]
        act = panel.visitors[stores][:, past]
        # What was on the books h days before each past day, the same lead as today's forecast.
        bk = K[stores][:, past, min(h, K.shape[2] - 1)]
        gain = np.where(np.isnan(act), np.nan, act - bk)
        with np.errstate(invalid="ignore"):
            n = np.sum(~np.isnan(gain), axis=1)
            mean_gain = np.where(n > 0, np.nansum(gain, axis=1) / np.maximum(n, 1), np.nan)
        out[:, h - 1] = np.clip(booked_now[:, h - 1] + mean_gain, 0, None)
    return out
