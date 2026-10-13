"""Feature-group ablation: what does each group of inputs actually buy?

Retrains the point model at every evaluation cutoff with one group of
features removed, using the same walk-forward protocol as the backtest, and
reports the change in RMSLE against the full model, overall and on the
slices where that group should matter. The store-level offset (the store's
own same-weekday level, see model.py) is kept in every variant, so "no
same-weekday history" means the trees lose those features, not the offset.

    .venv/bin/python scripts/ablate.py [--data-dir artifacts/raw] [--out artifacts/results/ablation]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from covercast.backtest import rmsle, training_cutoffs
from covercast.config import RAW_DIR, RESULTS_DIR
from covercast.data import load_panel
from covercast.features import CATEGORICAL, FEATURES, build_cutoff, cutoff_indices
from covercast.model import PARAMS, level

GROUPS = {
    "bookings": ["res_known", "res_parties_known", "res_typical_at_lead", "res_share_28"],
    "holidays": ["holiday", "holiday_prev", "holiday_next"],
    "calendar (month, day of month)": ["month", "dom"],
    "same-weekday history": ["sd_last", "sd_mean4", "sd_mean8", "sd_median8", "sd_open_frac13", "dow_effect"],
    "recent level and volatility": ["mean_7", "mean_28", "mean_91", "std_28", "open_frac_28"],
    "store type, area, age": ["genre", "pref", "store_age"],
    "lead time": ["h"],
}


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, feats: list[str]) -> np.ndarray:
    cats = [c for c in CATEGORICAL if c in feats]
    m = HistGradientBoostingRegressor(categorical_features=cats or None, **PARAMS)
    m.fit(train[feats].astype(float), np.log1p(train["actual"].to_numpy()) - level(train))
    return np.clip(np.expm1(m.predict(test[feats].astype(float)) + level(test)), 0, None)


def store_boot(df: pd.DataFrame, a: str, b: str, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    la, lb, y = np.log1p(df[a]), np.log1p(df[b]), np.log1p(df["actual"])
    g = df.assign(ea=(la - y) ** 2, eb=(lb - y) ** 2).groupby("store").agg(ea=("ea", "sum"), eb=("eb", "sum"), n=("ea", "size"))
    idx = np.random.default_rng(seed).integers(0, len(g), (n_boot, len(g)))
    n = g["n"].to_numpy()[idx].sum(1)
    d = np.sqrt(g["ea"].to_numpy()[idx].sum(1) / n) - np.sqrt(g["eb"].to_numpy()[idx].sum(1) / n)
    point = rmsle(df[a], df["actual"]) - rmsle(df[b], df["actual"])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "ablation")
    ap.add_argument("--n-eval", type=int, default=8)
    args = ap.parse_args()

    panel = load_panel(args.data_dir)
    all_ci, eval_ci = cutoff_indices(panel, args.n_eval)
    frames = {c: build_cutoff(panel, c) for c in all_ci}
    variants = {"full": FEATURES} | {f"without {g}": [f for f in FEATURES if f not in cols] for g, cols in GROUPS.items()}

    out = []
    for ci in eval_ci:
        train = pd.concat([frames[c] for c in training_cutoffs(all_ci, ci)], ignore_index=True)
        train = train[train["actual"].notna()]
        test = frames[ci][frames[ci]["actual"].notna()].copy()
        t0 = time.time()
        for name, feats in variants.items():
            test[name] = fit_predict(train, test, feats)
        out.append(test)
        print(f"cutoff {panel.dates[ci].date()}: {len(variants)} variants in {time.time() - t0:.0f}s", flush=True)
    pred = pd.concat(out, ignore_index=True)

    books = pred["res_share_28"] > 0
    near_hol = (pred["holiday"] + pred["holiday_prev"] + pred["holiday_next"]) > 0
    slices = {
        "all": np.ones(len(pred), bool),
        "stores that take bookings": books.to_numpy(),
        "holiday or next to one": near_hol.to_numpy(),
        "1-3 days ahead": (pred["h"] <= 3).to_numpy(),
        "8-14 days ahead": (pred["h"] >= 8).to_numpy(),
    }
    rows = []
    for name in variants:
        r = {"variant": name}
        for sname, m in slices.items():
            r[sname] = rmsle(pred.loc[m, name], pred.loc[m, "actual"])
        if name != "full":
            d, lo, hi = store_boot(pred, name, "full")
            r["delta_vs_full"], r["ci_low"], r["ci_high"] = d, lo, hi
        rows.append(r)
    tab = pd.DataFrame(rows).set_index("variant")
    tab["mean4 baseline, all"] = rmsle(pred["mean4"], pred["actual"])

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ablation.json").write_text(json.dumps(tab.round(5).to_dict(orient="index"), indent=2))
    md = ["# Feature-group ablation", "",
          f"{pred['cutoff'].nunique()} walk-forward cutoffs, {len(pred):,} store-days. RMSLE (lower is better); "
          "delta_vs_full > 0 means removing the group hurts, with a 95% store-bootstrap interval.", "",
          tab.round(4).to_markdown(), ""]
    (args.out / "ablation.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
