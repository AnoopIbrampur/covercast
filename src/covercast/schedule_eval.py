"""Shift-level backtest: build each policy's schedule for every store-day, replay the real day.

Reads the predictions and calibration a `covercast backtest` run saved, so the
forecasts are exactly the walk-forward ones; nothing here is refit.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np
import pandas as pd

from . import distribution as dist
from .data import Panel
from .schedule import ScheduleParams, arrival_profiles, open_hours, replay_cost, solve, staff_need

POLICIES = {
    # name: (forecast column or distribution, description)
    "mean4_padded": "4-week average + best padding, one scenario",
    "model_padded": "model mean + best padding, one scenario",
    "mean4_dist": "optimised over a no-model distribution around the 4-week average",
    "model_dist": "optimised over the model's calibrated distribution",
    "oracle": "perfect foresight of the realized hourly need",
}
VOLUME_BANDS = [0, 15, 30, 60, np.inf]
VOLUME_LABELS = ["<15 covers/day", "15-30", "30-60", "60+"]


def _seed(store: int, day: int) -> int:
    return (store * 1_000_003 + day * 7_919) % (2**32)


def _scenarios(total_quantiles: np.ndarray, profile: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Daily totals at stratified quantile levels, each split into hours multinomially."""
    totals = np.round(total_quantiles).astype(int)
    return rng.multinomial(totals, profile).astype(float)


def _work(args: tuple) -> list[dict]:
    rows, profiles, opens, z_model, z_base, params = args
    K = params.n_scenarios
    levels = (np.arange(K) + 0.5) / K
    out = []
    for r in rows:
        prof, om = profiles[r["store"]], opens[r["store"]]
        realized = np.random.default_rng(_seed(r["store"], r["day"])).multinomial(int(r["actual"]), prof).astype(float)
        need_real = staff_need(realized, params)
        scen_rng = np.random.default_rng(_seed(r["store"], r["day"]) + 1)

        model_q = dist.quantiles(np.array([r["center_log"]]), np.array([r["scale_log"]]), z_model, levels)[0]
        base_q = dist.quantiles(np.array([np.log1p(r["mean4"])]), np.ones(1), z_base, levels)[0]
        needs = {
            "mean4_padded": staff_need(r["mean4_padded"] * prof, params)[None],
            "model_padded": staff_need(r["model_padded"] * prof, params)[None],
            "mean4_dist": staff_need(_scenarios(base_q, prof, scen_rng), params),
            "model_dist": staff_need(_scenarios(model_q, prof, scen_rng), params),
            "oracle": need_real[None],
        }
        rec = {"store": r["store"], "day": r["day"], "h": r["h"]}
        for name, need in needs.items():
            floor = solve(need, om, params)
            cost, hours, short = replay_cost(floor, need_real, params)
            rec[f"cost_{name}"], rec[f"hours_{name}"], rec[f"short_{name}"] = cost, hours, short
        out.append(rec)
    return out


def run(
    panel: Panel,
    pred: pd.DataFrame,
    calibration: list[dict],
    params: ScheduleParams | None = None,
    n_stores: int | None = None,
    workers: int | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    params = params or ScheduleParams()
    pred = pred.copy()
    if n_stores is not None and n_stores < pred["store"].nunique():
        keep = np.random.default_rng(seed).choice(pred["store"].unique(), n_stores, replace=False)
        pred = pred[pred["store"].isin(keep)]
    cal = {c["cutoff"]: c for c in calibration}
    workers = workers or max(1, (os.cpu_count() or 2) - 1)

    jobs = []
    for cutoff, g in pred.groupby("cutoff"):
        ci = panel.dates.get_loc(pd.Timestamp(cutoff))
        prof = arrival_profiles(panel, ci, params)
        stores = g["store"].unique()
        profiles = {int(s): prof.arrival[s] for s in stores}
        opens = {int(s): open_hours(prof.arrival[s], params) for s in stores}
        c = cal[pd.Timestamp(cutoff).date().isoformat()]
        recs = [
            {
                "store": int(row.store),
                "day": int((row.target_date - panel.dates[0]).days),
                "h": int(row.h),
                "actual": row.actual,
                "center_log": row.center_log,
                "scale_log": row.scale_log,
                "mean4": row.mean4,
                "mean4_padded": row.mean4_padded,
                "model_padded": row.model_padded,
            }
            for row in g.itertuples()
        ]
        for i in range(0, len(recs), 400):
            jobs.append((recs[i : i + 400], profiles, opens, np.array(c["model_z"]), np.array(c["mean4_z"]), params))

    if verbose:
        print(f"scheduling {len(pred):,} store-days x {len(POLICIES)} policies on {workers} workers")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = [r for chunk in ex.map(_work, jobs) for r in chunk]
    out = pd.DataFrame(results)
    meta = pred.assign(day=(pred["target_date"] - panel.dates[0]).dt.days)[
        ["store", "day", "cutoff", "target_date", "h", "actual", "mean_28", "res_share_28"]
    ]
    # A day is forecast at two cutoffs (1-7 and 8-14 days ahead), so the lead time is part of the key.
    return meta.merge(out, on=["store", "day", "h"], validate="one_to_one")


def summary_table(sched: pd.DataFrame, params: ScheduleParams) -> pd.DataFrame:
    """Per store-week (7 open store-days): wages, short hours, total cost, and excess over perfect foresight."""
    weeks = len(sched) / 7
    rows = []
    for p in POLICIES:
        rows.append(
            {
                "policy": p,
                "total_cost": sched[f"cost_{p}"].sum() / weeks,
                "scheduled_hours": sched[f"hours_{p}"].sum() / weeks,
                "short_staff_hours": sched[f"short_{p}"].sum() / weeks,
                "excess_over_oracle": (sched[f"cost_{p}"] - sched["cost_oracle"]).sum() / weeks,
            }
        )
    return pd.DataFrame(rows).set_index("policy")


def by_volume(sched: pd.DataFrame) -> pd.DataFrame:
    """Excess cost over perfect foresight per store-week, by how busy the store was going in."""
    band = pd.cut(np.expm1(sched["mean_28"]), VOLUME_BANDS, labels=VOLUME_LABELS, right=False)
    rows = []
    for label, g in sched.groupby(band, observed=True):
        weeks = len(g) / 7
        rec = {"band": label, "store_days": len(g), "stores": g["store"].nunique()}
        for p in ("mean4_padded", "model_padded", "mean4_dist", "model_dist"):
            rec[p] = (g[f"cost_{p}"] - g["cost_oracle"]).sum() / weeks
        base = (g["cost_mean4_padded"] - g["cost_oracle"]).sum()
        rec["model_dist_cuts_excess_by"] = 1 - (g["cost_model_dist"] - g["cost_oracle"]).sum() / base if base > 0 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows).set_index("band")


def params_dict(params: ScheduleParams) -> dict:
    return asdict(params)
