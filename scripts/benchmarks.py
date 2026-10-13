"""Benchmarks against the gradient-boosted model, on the backtest's own test weeks.

1. Weekday exponential smoothing, its three parameters tuned on the 4 cutoffs
   just before the test window (never on test weeks).
2. Booking pickup, alone, on stores that take bookings.
3. The boosted model with the pickup forecast added as an input.

Reads the boosted model's walk-forward forecasts from a finished backtest
(artifacts/results/recruit/predictions.csv.gz), so the comparison is on
exactly the same store-days.

    .venv/bin/python scripts/benchmarks.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from covercast.backtest import rmsle, training_cutoffs
from covercast.classical import booked_by_lead, pickup, weekday_smoothing
from covercast.config import HORIZON, RAW_DIR, RESULTS_DIR
from covercast.data import load_panel
from covercast.features import CATEGORICAL, FEATURES, build_cutoff, cutoff_indices
from covercast.model import PARAMS, level

GRID = {"alpha": [0.05, 0.1, 0.2, 0.3], "alpha_dev": [0.05, 0.1, 0.2, 0.4], "phi": [0.5, 0.8, 0.9, 0.97]}


def attach(frame: pd.DataFrame, stores: np.ndarray, fc: np.ndarray, name: str) -> pd.DataFrame:
    """fc is (len(stores), HORIZON); map onto the frame's (store, h) rows."""
    pos = {s: i for i, s in enumerate(stores)}
    i = frame["store"].map(pos).to_numpy()
    out = frame.copy()
    out[name] = fc[i, frame["h"].to_numpy() - 1]
    return out


def smoothing_frame(panel, frames, c, params) -> pd.DataFrame:
    f = frames[c][frames[c]["actual"].notna()]
    stores = np.sort(f["store"].unique())
    return attach(f, stores, weekday_smoothing(panel, c, stores, **params), "smooth")


def pickup_frame(panel, K, frame: pd.DataFrame, c: int) -> pd.DataFrame:
    stores = np.sort(frame["store"].unique())
    booked = np.zeros((len(stores), HORIZON))
    sub = frame.drop_duplicates(["store", "h"]).set_index(["store", "h"])["res_known"]
    for k, s in enumerate(stores):
        for h in range(1, HORIZON + 1):
            booked[k, h - 1] = np.expm1(sub.get((s, h), 0.0))
    return attach(frame, stores, pickup(panel, K, c, stores, booked), "pickup")


def store_boot(df, a, b, n_boot=1000, seed=0):
    y = np.log1p(df["actual"])
    g = df.assign(ea=(np.log1p(df[a]) - y) ** 2, eb=(np.log1p(df[b]) - y) ** 2).groupby("store").agg(
        ea=("ea", "sum"), eb=("eb", "sum"), n=("ea", "size"))
    idx = np.random.default_rng(seed).integers(0, len(g), (n_boot, len(g)))
    n = g["n"].to_numpy()[idx].sum(1)
    d = np.sqrt(g["ea"].to_numpy()[idx].sum(1) / n) - np.sqrt(g["eb"].to_numpy()[idx].sum(1) / n)
    return float(rmsle(df[a], df["actual"]) - rmsle(df[b], df["actual"])), *map(float, np.percentile(d, [2.5, 97.5]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--results", type=Path, default=RESULTS_DIR / "recruit")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "benchmarks")
    args = ap.parse_args()

    panel = load_panel(args.data_dir)
    all_ci, eval_ci = cutoff_indices(panel, 8)
    frames = {c: build_cutoff(panel, c) for c in all_ci}
    val_ci = [c for c in all_ci if c + HORIZON <= eval_ci[0]][-4:]

    # 1. Tune the smoothing model on validation cutoffs only.
    t0 = time.time()
    scores = []
    for a, ad, phi in itertools.product(GRID["alpha"], GRID["alpha_dev"], GRID["phi"]):
        prm = dict(alpha=a, alpha_dev=ad, phi=phi)
        v = pd.concat([smoothing_frame(panel, frames, c, prm) for c in val_ci])
        scores.append((rmsle(v["smooth"], v["actual"]), prm))
    scores.sort(key=lambda r: r[0])
    best_val, best = scores[0]
    print(f"smoothing tuned on {len(val_ci)} validation cutoffs in {time.time() - t0:.0f}s: {best}, val RMSLE {best_val:.4f}", flush=True)

    # 2-3. Pickup for every cutoff (as a feature it's needed in training too).
    K = booked_by_lead(panel)
    pk = {c: pickup_frame(panel, K, frames[c], c) for c in all_ci}
    for c in all_ci:
        f = pk[c]
        f["pickup_rel"] = np.where(f["res_share_28"] > 0, np.log1p(f["pickup"]) - level(f), np.nan)
    feats = FEATURES + ["pickup_rel"]

    pred = pd.read_csv(args.results / "predictions.csv.gz", parse_dates=["cutoff", "target_date"])[
        ["cutoff", "store", "h", "model"]]
    rows = []
    for c in eval_ci:
        test = pk[c][pk[c]["actual"].notna()].copy()
        s = smoothing_frame(panel, frames, c, best)[["store", "h", "smooth"]]
        test = test.merge(s, on=["store", "h"])
        train = pd.concat([pk[t] for t in training_cutoffs(all_ci, c)], ignore_index=True)
        train = train[train["actual"].notna()]
        m = HistGradientBoostingRegressor(categorical_features=CATEGORICAL, **PARAMS)
        m.fit(train[feats].astype(float), np.log1p(train["actual"].to_numpy()) - level(train))
        test["model_pickup"] = np.clip(np.expm1(m.predict(test[feats].astype(float)) + level(test)), 0, None)
        rows.append(test)
        print(f"cutoff {panel.dates[c].date()} done", flush=True)
    t = pd.concat(rows, ignore_index=True)
    t = t.merge(pred.assign(cutoff=pred["cutoff"]), on=["cutoff", "store", "h"], how="inner")

    lv = np.expm1(level(t))
    slices = {
        "all": np.ones(len(t), bool),
        "excluding collapse days (<35% of usual)": (t["actual"] >= 0.35 * lv).to_numpy(),
        "stores that take bookings": (t["res_share_28"] > 0).to_numpy(),
    }
    methods = ["mean4", "smooth", "model", "model_pickup", "pickup"]
    tab = {}
    for name, msk in slices.items():
        g = t[msk]
        tab[name] = {mth: rmsle(g[mth].fillna(g["mean4"]), g["actual"]) for mth in methods}
    tab = pd.DataFrame(tab).T
    bk = t[t["res_share_28"] > 0]
    comps = {
        "model vs smoothing": store_boot(t, "model", "smooth"),
        "model+pickup vs model": store_boot(t, "model_pickup", "model"),
        "pickup vs model (booking stores)": store_boot(bk.assign(pickup=bk["pickup"].fillna(bk["mean4"])), "pickup", "model"),
        "model+pickup vs model (booking stores)": store_boot(bk, "model_pickup", "model"),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "benchmarks.json").write_text(json.dumps({"smoothing_params": best, "smoothing_val_rmsle": best_val,
        "rmsle": tab.round(5).to_dict(orient="index"), "comparisons": comps}, indent=2))
    md = ["# Benchmarks", "",
          f"Same 8 walk-forward test weeks as the backtest, {len(t):,} store-days. Smoothing tuned on the 4 cutoffs before them: {best}.",
          "Pickup applies only to stores that take bookings (elsewhere it falls back to the 4-week average).", "",
          tab.round(4).to_markdown(), "", "| comparison | RMSLE difference | 95% interval |", "|---|---:|---:|"]
    md += [f"| {k} | {v[0]:+.4f} | {v[1]:+.4f} to {v[2]:+.4f} |" for k, v in comps.items()]
    (args.out / "benchmarks.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
