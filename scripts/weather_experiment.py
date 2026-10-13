"""Does weather help the forecast?

The weather here is observed (Open-Meteo archive), not forecast, because
2016-17 weather forecasts aren't publicly archived. So two variants:

- **ceiling**: the target day's actual weather at every lead time, i.e. a
  perfect weather forecast. If this barely helps, weather isn't worth pursuing.
- **realistic**: the same, but only 1-3 days ahead, where real weather
  forecasts are skilful; beyond that the model gets nothing.

Both are trained the same way they're tested (the realistic variant masks
h > 3 in training too), walk-forward on the backtest's 8 test weeks.

    .venv/bin/python -m covercast.weather            # once, fetches artifacts/raw/weather.csv
    .venv/bin/python scripts/weather_experiment.py
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
from covercast.weather import store_weather

WEATHER = ["w_precip", "w_evening_precip", "w_snow", "w_tmax", "w_temp_anom"]
REALISTIC_MAX_H = 3


def add_weather(frame: pd.DataFrame, panel, W: dict[str, np.ndarray]) -> pd.DataFrame:
    f = frame.copy()
    ti = (f["target_date"] - panel.dates[0]).dt.days.to_numpy()
    ok = ti < panel.n_days
    s = f["store"].to_numpy()
    for col, src in zip(WEATHER, ["precip", "evening_precip", "snow", "tmax", "temp_anom"]):
        v = np.full(len(f), np.nan)
        v[ok] = W[src][s[ok], ti[ok]]
        f[col] = v
    return f


def realistic(f: pd.DataFrame) -> pd.DataFrame:
    f = f.copy()
    f.loc[f["h"] > REALISTIC_MAX_H, WEATHER] = np.nan
    return f


def fit_predict(train, test, feats):
    m = HistGradientBoostingRegressor(categorical_features=CATEGORICAL, **PARAMS)
    m.fit(train[feats].astype(float), np.log1p(train["actual"].to_numpy()) - level(train))
    return np.clip(np.expm1(m.predict(test[feats].astype(float)) + level(test)), 0, None)


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
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "weather")
    args = ap.parse_args()

    panel = load_panel(args.data_dir)
    W = store_weather(args.data_dir, panel.store_ids, panel.dates)
    print(f"weather coverage: {np.isfinite(W['precip']).mean():.1%} of store-days", flush=True)
    all_ci, eval_ci = cutoff_indices(panel, 8)
    frames = {c: add_weather(build_cutoff(panel, c), panel, W) for c in all_ci}

    # Does weather move demand at all? Covers vs the store's usual level, on all observed history.
    hist = pd.concat([frames[c] for c in all_ci if c < eval_ci[0]], ignore_index=True)
    hist = hist[hist["actual"].notna() & (hist["h"] == 1)]
    hist["ratio"] = hist["actual"] / np.expm1(level(hist))
    bins = {"dry (<1 mm)": hist["w_precip"] < 1, "rain 1-10 mm": hist["w_precip"].between(1, 10),
            "rain >10 mm": hist["w_precip"] > 10, "evening rain >=3 mm": hist["w_evening_precip"] >= 3,
            "snow": hist["w_snow"] > 0, "much colder than usual (<-5C)": hist["w_temp_anom"] < -5,
            "much warmer than usual (>+5C)": hist["w_temp_anom"] > 5}
    effect = pd.DataFrame({k: {"days": int(m.sum()), "median covers vs usual": float(hist.loc[m, "ratio"].median())}
                           for k, m in bins.items()}).T

    out = []
    for c in eval_ci:
        tr = pd.concat([frames[t] for t in training_cutoffs(all_ci, c)], ignore_index=True)
        tr = tr[tr["actual"].notna()]
        te = frames[c][frames[c]["actual"].notna()].copy()
        t0 = time.time()
        te["base"] = fit_predict(tr, te, FEATURES)
        te["ceiling"] = fit_predict(tr, te, FEATURES + WEATHER)
        te["realistic"] = fit_predict(realistic(tr), realistic(te), FEATURES + WEATHER)
        out.append(te)
        print(f"cutoff {panel.dates[c].date()}: {time.time() - t0:.0f}s", flush=True)
    t = pd.concat(out, ignore_index=True)

    slices = {"all": np.ones(len(t), bool), "1-3 days ahead": (t["h"] <= 3).to_numpy(),
              "rainy days (>=5 mm)": (t["w_precip"] >= 5).to_numpy(),
              "rainy evenings (>=3 mm, 17-22h)": (t["w_evening_precip"] >= 3).to_numpy(),
              "unusual temperature (|anomaly| > 5C)": (t["w_temp_anom"].abs() > 5).to_numpy()}
    tab = pd.DataFrame({k: {"store_days": int(m.sum()), **{v: rmsle(t.loc[m, v], t.loc[m, "actual"]) for v in ("base", "ceiling", "realistic")}}
                        for k, m in slices.items()}).T
    comps = {"ceiling vs base": store_boot(t, "ceiling", "base"),
             "realistic vs base": store_boot(t, "realistic", "base"),
             "realistic vs base, 1-3 days ahead": store_boot(t[t["h"] <= 3], "realistic", "base"),
             "ceiling vs base, rainy days": store_boot(t[t["w_precip"] >= 5], "ceiling", "base")}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "weather.json").write_text(json.dumps({"effect": effect.to_dict(orient="index"),
        "rmsle": tab.to_dict(orient="index"), "comparisons": comps}, indent=2, default=float))
    md = ["# Weather experiment", "",
          "Observed weather from the Open-Meteo archive. *ceiling* = target-day weather at every lead time (a perfect "
          f"weather forecast); *realistic* = only {REALISTIC_MAX_H} or fewer days ahead.", "",
          "## Does weather move demand? (covers relative to the store's usual level, pre-test history)", "",
          effect.round(3).to_markdown(), "", "## Forecast accuracy (RMSLE)", "", tab.round(4).to_markdown(), "",
          "| comparison | RMSLE difference | 95% interval |", "|---|---:|---:|"]
    md += [f"| {k} | {v[0]:+.4f} | {v[1]:+.4f} to {v[2]:+.4f} |" for k, v in comps.items()]
    (args.out / "weather.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
