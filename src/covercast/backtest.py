"""Walk-forward evaluation: retrain at every weekly cutoff, forecast the next 14 days.

A model evaluated at cutoff c is trained only on earlier cutoffs whose whole
14-day horizon had already happened by c, so no training target postdates the
forecast it is judged against.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import model as mdl
from .config import HORIZON, StaffingCosts
from .data import Panel
from .features import build_cutoff, cutoff_indices
from .staffing import best_multiplier, hours_over_under, labor_cost

# Forecasts judged on accuracy.
POINT_METHODS = ["last_week", "mean4", "model"]
# Forecasts judged on staffing cost: each baseline with its best padding, the
# model's mean forecast with its best padding, and the model asked directly
# for the cost-minimising quantile.
STAFF_METHODS = ["last_week_padded", "mean4_padded", "model_padded", "model_quantile"]


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    costs: StaffingCosts
    multipliers: list[dict] = field(default_factory=list)
    n_train_rows: list[int] = field(default_factory=list)


def run(panel: Panel, n_eval: int = 8, costs: StaffingCosts | None = None, verbose: bool = True) -> BacktestResult:
    costs = costs or StaffingCosts()
    all_ci, eval_ci = cutoff_indices(panel, n_eval)
    t0 = time.time()
    frames = {ci: build_cutoff(panel, ci) for ci in all_ci}
    if verbose:
        print(f"built features for {len(all_ci)} cutoffs in {time.time() - t0:.1f}s")

    out, mults, n_rows = [], [], []
    for ci in eval_ci:
        train = pd.concat([frames[c] for c in training_cutoffs(all_ci, ci)], ignore_index=True)
        train = train[train["actual"].notna()]
        test = frames[ci][frames[ci]["actual"].notna()].copy()
        if train.empty or test.empty:
            continue

        t1 = time.time()
        point = mdl.fit(train)
        quant = mdl.fit(train, quantile=costs.critical_ratio)
        # In-sample model forecasts would be overfit, so the model's padding is
        # tuned on its last eight training cutoffs using a model that never saw them.
        model_k = _model_padding(train, costs)

        test["model"] = mdl.predict(point, test)
        test["model_quantile"] = mdl.predict(quant, test)
        k = {b: best_multiplier(train[b], train["actual"], costs) for b in ("last_week", "mean4")}
        k["model"] = model_k
        for b in ("last_week", "mean4", "model"):
            test[f"{b}_padded"] = k[b] * test[b]
        out.append(test)
        mults.append({"cutoff": panel.dates[ci].date().isoformat(), **k})
        n_rows.append(len(train))
        if verbose:
            print(
                f"cutoff {panel.dates[ci].date()}: train {len(train):,} rows, "
                f"test {len(test):,}, padding {k}, {time.time() - t1:.1f}s"
            )

    return BacktestResult(pd.concat(out, ignore_index=True), costs, mults, n_rows)


def training_cutoffs(all_ci: list[int], eval_ci: int, horizon: int = HORIZON) -> list[int]:
    """Earlier cutoffs whose every target day is on or before the evaluation cutoff."""
    return [c for c in all_ci if c + horizon <= eval_ci]


def _model_padding(train: pd.DataFrame, costs: StaffingCosts, n_holdout: int = 8) -> float:
    cutoffs = np.sort(train["cutoff"].unique())
    if len(cutoffs) <= n_holdout + 4:
        return 1.0
    split = cutoffs[-n_holdout]
    # Same rule as the outer loop: an inner training row's whole horizon must
    # be over before the holdout starts.
    inner = train[train["cutoff"] + pd.Timedelta(days=HORIZON) <= split]
    hold = train[train["cutoff"] >= split]
    if inner.empty or hold.empty:
        return 1.0
    return best_multiplier(mdl.predict(mdl.fit(inner), hold), hold["actual"], costs)


def rmsle(f: np.ndarray, a: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.log1p(f) - np.log1p(a)) ** 2)))


def accuracy_table(pred: pd.DataFrame) -> pd.DataFrame:
    a = pred["actual"].to_numpy()
    rows = []
    for m in POINT_METHODS:
        f = pred[m].to_numpy()
        rows.append(
            {
                "method": m,
                "rmsle": rmsle(f, a),
                "mae_covers": float(np.mean(np.abs(f - a))),
                "wape": float(np.sum(np.abs(f - a)) / np.sum(a)),
                "bias": float(np.sum(f - a) / np.sum(a)),
            }
        )
    return pd.DataFrame(rows).set_index("method")


def staffing_table(pred: pd.DataFrame, costs: StaffingCosts) -> pd.DataFrame:
    a = pred["actual"].to_numpy()
    # A store-week here is 7 open store-days, the same unit store_bootstrap uses.
    # Closed days cost nothing whatever the forecast, so they are not counted.
    store_weeks = len(pred) / 7
    rows = []
    for m in POINT_METHODS + STAFF_METHODS:
        f = pred[m].to_numpy()
        over, under = hours_over_under(f, a, costs)
        rows.append(
            {
                "method": m,
                "cost_per_store_week": float(labor_cost(f, a, costs).sum() / store_weeks),
                "idle_hours_per_store_week": float(over.sum() / store_weeks),
                "short_hours_per_store_week": float(under.sum() / store_weeks),
                "short_day_rate": float(np.mean(f < a)),
            }
        )
    return pd.DataFrame(rows).set_index("method")


def by_horizon(pred: pd.DataFrame) -> pd.DataFrame:
    return pred.groupby("h").apply(
        lambda g: pd.Series({m: rmsle(g[m], g["actual"]) for m in POINT_METHODS}), include_groups=False
    )


def per_cutoff_wins(pred: pd.DataFrame, a: str, b: str, costs: StaffingCosts | None = None) -> tuple[int, int]:
    """How many evaluation cutoffs `a` beats `b` on (RMSLE, or labor cost if costs given)."""
    wins = 0
    groups = pred.groupby("cutoff")
    for _, g in groups:
        if costs is None:
            wins += rmsle(g[a], g["actual"]) < rmsle(g[b], g["actual"])
        else:
            wins += labor_cost(g[a], g["actual"], costs).sum() < labor_cost(g[b], g["actual"], costs).sum()
    return int(wins), len(groups)


def store_bootstrap(
    pred: pd.DataFrame, a: str, b: str, costs: StaffingCosts | None = None, n_boot: int = 2000, seed: int = 0
) -> dict[str, float]:
    """Store-level block bootstrap of (a - b): RMSLE difference, or labor-cost difference per store-week.

    Rows from one store are correlated (same kitchen, same neighbourhood), so
    resampling stores rather than rows gives an honest interval.
    """
    g = pred.assign(
        ea=(np.log1p(pred[a]) - np.log1p(pred["actual"])) ** 2,
        eb=(np.log1p(pred[b]) - np.log1p(pred["actual"])) ** 2,
    )
    if costs is not None:
        g["ca"] = labor_cost(pred[a], pred["actual"], costs)
        g["cb"] = labor_cost(pred[b], pred["actual"], costs)
    agg = g.groupby("store").agg(
        n=("actual", "size"), ea=("ea", "sum"), eb=("eb", "sum"), **({"ca": ("ca", "sum"), "cb": ("cb", "sum")} if costs else {})
    )
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(agg), size=(n_boot, len(agg)))
    n = agg["n"].to_numpy()[idx].sum(axis=1)
    if costs is None:
        stat = lambda ia: np.sqrt(agg["ea"].to_numpy()[ia].sum(-1) / n) - np.sqrt(agg["eb"].to_numpy()[ia].sum(-1) / n)
        point = rmsle(pred[a], pred["actual"]) - rmsle(pred[b], pred["actual"])
    else:
        # Store-days / 7 = store-weeks.
        stat = lambda ia: (agg["ca"].to_numpy()[ia].sum(-1) - agg["cb"].to_numpy()[ia].sum(-1)) / (n / 7)
        point = float((g["ca"].sum() - g["cb"].sum()) / (len(g) / 7))
    draws = stat(idx)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"diff": float(point), "ci_low": float(lo), "ci_high": float(hi)}
