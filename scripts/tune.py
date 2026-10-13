"""Hyperparameter search for the point model, without touching the test weeks.

Random search over the boosted model's main knobs. Each candidate is scored
walk-forward on the 4 cutoffs just before the test window (each refit on its
own past), the best is chosen on that score alone, and only then are the
default and the chosen settings compared on the 8 test weeks.

    .venv/bin/python scripts/tune.py [--trials 16]
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
from covercast.config import HORIZON, RAW_DIR, RESULTS_DIR
from covercast.data import load_panel
from covercast.features import CATEGORICAL, FEATURES, build_cutoff, cutoff_indices
from covercast.model import PARAMS, level

SPACE = {
    "learning_rate": [0.02, 0.03, 0.05, 0.08],
    "max_iter": [300, 500, 800],
    "max_leaf_nodes": [15, 31, 63, 127],
    "min_samples_leaf": [20, 50, 100, 300],
    "l2_regularization": [0.0, 1.0, 5.0, 20.0],
    "max_features": [0.5, 0.8, 1.0],
}


def score(frames, all_ci, cutoffs, params) -> tuple[float, pd.DataFrame]:
    out = []
    for c in cutoffs:
        tr = pd.concat([frames[t] for t in training_cutoffs(all_ci, c)], ignore_index=True)
        tr = tr[tr["actual"].notna()]
        te = frames[c][frames[c]["actual"].notna()].copy()
        m = HistGradientBoostingRegressor(categorical_features=CATEGORICAL, **params)
        m.fit(tr[FEATURES].astype(float), np.log1p(tr["actual"].to_numpy()) - level(tr))
        te["fc"] = np.clip(np.expm1(m.predict(te[FEATURES].astype(float)) + level(te)), 0, None)
        out.append(te)
    df = pd.concat(out, ignore_index=True)
    return rmsle(df["fc"], df["actual"]), df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "tuning")
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    panel = load_panel(args.data_dir)
    all_ci, eval_ci = cutoff_indices(panel, 8)
    frames = {c: build_cutoff(panel, c) for c in all_ci}
    val_ci = [c for c in all_ci if c + HORIZON <= eval_ci[0]][-4:]

    rng = np.random.default_rng(args.seed)
    base = {k: PARAMS[k] for k in ("early_stopping", "random_state")}
    trials = [dict(PARAMS)] + [base | {k: v[rng.integers(len(v))] for k, v in SPACE.items()} for _ in range(args.trials)]
    results = []
    for i, prm in enumerate(trials):
        t0 = time.time()
        s, _ = score(frames, all_ci, val_ci, prm)
        results.append({"trial": i, "val_rmsle": s, **{k: prm.get(k) for k in SPACE}})
        print(f"trial {i:2d} {'(default)' if i == 0 else ''} val RMSLE {s:.4f} in {time.time() - t0:.0f}s {results[-1]}", flush=True)
    res = pd.DataFrame(results).sort_values("val_rmsle")
    best = trials[int(res.iloc[0]["trial"])]

    test_default, d_def = score(frames, all_ci, eval_ci, PARAMS)
    test_best, d_best = score(frames, all_ci, eval_ci, best)
    d = d_def[["store", "cutoff", "h", "actual", "fc"]].merge(
        d_best[["store", "cutoff", "h", "fc"]], on=["store", "cutoff", "h"], suffixes=("_default", "_tuned"))
    y = np.log1p(d["actual"])
    g = d.assign(a=(np.log1p(d["fc_tuned"]) - y) ** 2, b=(np.log1p(d["fc_default"]) - y) ** 2).groupby("store").agg(
        a=("a", "sum"), b=("b", "sum"), n=("a", "size"))
    idx = np.random.default_rng(0).integers(0, len(g), (1000, len(g)))
    n = g["n"].to_numpy()[idx].sum(1)
    boot = np.sqrt(g["a"].to_numpy()[idx].sum(1) / n) - np.sqrt(g["b"].to_numpy()[idx].sum(1) / n)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"validation_cutoffs": [str(panel.dates[c].date()) for c in val_ci],
               "default_val_rmsle": float(res.loc[res.trial == 0, "val_rmsle"].iloc[0]),
               "best": {k: best.get(k) for k in SPACE}, "best_val_rmsle": float(res.iloc[0]["val_rmsle"]),
               "test_rmsle_default": test_default, "test_rmsle_tuned": test_best,
               "test_diff": test_best - test_default, "test_diff_ci": [float(lo), float(hi)]}
    (args.out / "tuning.json").write_text(json.dumps(summary, indent=2, default=float))
    md = ["# Hyperparameter search", "",
          f"{len(trials)} settings (default + {args.trials} random), each scored walk-forward on validation cutoffs "
          f"{', '.join(summary['validation_cutoffs'])}; none of these are test weeks.", "",
          res.round(4).to_markdown(index=False), "",
          f"On the 8 test weeks: default {test_default:.4f}, tuned {test_best:.4f}, difference {test_best - test_default:+.4f} "
          f"(95% store-bootstrap {lo:+.4f} to {hi:+.4f}).", ""]
    (args.out / "tuning.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
