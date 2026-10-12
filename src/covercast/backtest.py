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

from . import distribution as dist
from . import model as mdl
from .config import HORIZON, StaffingCosts
from .data import Panel
from .features import build_cutoff, cutoff_indices
from .staffing import best_multiplier, hours_over_under, labor_cost

# Forecasts judged on accuracy.
POINT_METHODS = ["last_week", "mean4", "model"]
# Forecasts judged on staffing cost: each baseline with its best padding, the
# model's mean forecast with its best padding, the model asked directly for
# the cost-minimising quantile, and that quantile read off the calibrated
# distribution (with a no-model distribution around the 4-week average for comparison).
STAFF_METHODS = ["last_week_padded", "mean4_padded", "model_padded", "model_quantile", "mean4_dist", "model_dist"]


# Calibration cutoffs per evaluation (half fit the scale model, half calibrate),
# and the least history a walk-forward model may be trained on.
N_HOLDOUT = 8
MIN_TRAIN_CUTOFFS = 4


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    costs: StaffingCosts
    multipliers: list[dict] = field(default_factory=list)
    n_train_rows: list[int] = field(default_factory=list)
    # Per evaluation cutoff: the z quantile tables for the model and the
    # baseline distributions. With center_log / scale_log in predictions, these
    # reproduce any quantile of any forecast.
    calibration: list[dict] = field(default_factory=list)


def run(panel: Panel, n_eval: int = 8, costs: StaffingCosts | None = None, verbose: bool = True) -> BacktestResult:
    costs = costs or StaffingCosts()
    all_ci, eval_ci = cutoff_indices(panel, n_eval)
    t0 = time.time()
    frames = {ci: build_cutoff(panel, ci) for ci in all_ci}
    if verbose:
        print(f"built features for {len(all_ci)} cutoffs in {time.time() - t0:.1f}s")

    # Every cutoff that is either evaluated or used to calibrate an evaluation
    # gets its own walk-forward point model, trained only on cutoffs whose
    # horizon ended by then. The same fit serves as the evaluation forecast at
    # one cutoff and as a calibration forecast for later ones.
    holdouts = {ci: training_cutoffs(all_ci, ci)[-N_HOLDOUT:] for ci in eval_ci}
    needed = sorted(set(eval_ci).union(*holdouts.values()))
    wf: dict[int, pd.DataFrame] = {}
    for c in needed:
        tc = training_cutoffs(all_ci, c)
        if len(tc) < MIN_TRAIN_CUTOFFS:
            continue
        train = pd.concat([frames[t] for t in tc], ignore_index=True)
        train = train[train["actual"].notna()]
        f = frames[c][frames[c]["actual"].notna()].copy()
        t1 = time.time()
        f["pred_log"] = mdl.predict_log(mdl.fit(train), f)
        wf[c] = f
        if verbose:
            print(f"walk-forward fit for {panel.dates[c].date()}: {len(train):,} rows, {time.time() - t1:.1f}s")

    out, mults, n_rows, calib = [], [], [], []
    for ci in eval_ci:
        if ci not in wf or not all(h in wf for h in holdouts[ci]) or len(holdouts[ci]) < N_HOLDOUT:
            continue
        train = pd.concat([frames[c] for c in training_cutoffs(all_ci, ci)], ignore_index=True)
        train = train[train["actual"].notna()]
        hold = pd.concat([wf[h] for h in holdouts[ci]], ignore_index=True)
        test = wf[ci].copy()

        t1 = time.time()
        cal = dist.calibrate(hold)
        base_z = dist.baseline_z_quantiles(hold)
        quant = mdl.fit(train, quantile=costs.critical_ratio)

        test["center_log"] = test.pop("pred_log")
        test["scale_log"] = cal.scale(test)
        test["model"] = np.clip(np.expm1(test["center_log"]), 0, None)
        test["model_quantile"] = mdl.predict(quant, test)
        q = costs.critical_ratio
        test["model_dist"] = dist.quantiles(test["center_log"].to_numpy(), test["scale_log"].to_numpy(), cal.z_quantiles, q)[:, 0]
        test["mean4_dist"] = dist.quantiles(np.log1p(test["mean4"].to_numpy()), np.ones(len(test)), base_z, q)[:, 0]

        # Padding is tuned on past data only: the baselines on all training
        # rows, the model on its holdout cutoffs' walk-forward forecasts.
        k = {b: best_multiplier(train[b], train["actual"], costs) for b in ("last_week", "mean4")}
        k["model"] = best_multiplier(cal.holdout_forecasts["forecast"], cal.holdout_forecasts["actual"], costs)
        for b in ("last_week", "mean4", "model"):
            test[f"{b}_padded"] = k[b] * test[b]
        out.append(test)
        cutoff = panel.dates[ci].date().isoformat()
        mults.append({"cutoff": cutoff, **k})
        calib.append({"cutoff": cutoff, "model_z": cal.z_quantiles.tolist(), "mean4_z": base_z.tolist()})
        n_rows.append(len(train))
        if verbose:
            print(
                f"cutoff {panel.dates[ci].date()}: train {len(train):,} rows, "
                f"test {len(test):,}, padding {k}, {time.time() - t1:.1f}s"
            )

    return BacktestResult(pd.concat(out, ignore_index=True), costs, mults, n_rows, calib)


def training_cutoffs(all_ci: list[int], eval_ci: int, horizon: int = HORIZON) -> list[int]:
    """Earlier cutoffs whose every target day is on or before the evaluation cutoff."""
    return [c for c in all_ci if c + horizon <= eval_ci]


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


def segment_table(pred: pd.DataFrame, costs: StaffingCosts) -> pd.DataFrame:
    """Where the gains come from: stores that take bookings vs not, holiday-adjacent days vs normal."""
    near_holiday = (pred["holiday"] + pred["holiday_prev"] + pred["holiday_next"]) > 0
    books = pred["res_share_28"] > 0
    segments = {
        "all": np.ones(len(pred), bool),
        "takes bookings": books,
        "no bookings": ~books,
        "holiday or next to one": near_holiday,
        "other days": ~near_holiday,
    }
    rows = []
    for name, m in segments.items():
        g = pred[np.asarray(m)]
        base = labor_cost(g["mean4_padded"], g["actual"], costs).sum()
        quant = labor_cost(g["model_quantile"], g["actual"], costs).sum()
        rows.append(
            {
                "segment": name,
                "store_days": len(g),
                "stores": g["store"].nunique(),
                "rmsle_mean4": rmsle(g["mean4"], g["actual"]),
                "rmsle_model": rmsle(g["model"], g["actual"]),
                "cost_mean4_padded": base / (len(g) / 7),
                "cost_model_quantile": quant / (len(g) / 7),
                "cost_saving": 1 - quant / base,
            }
        )
    return pd.DataFrame(rows).set_index("segment")


DIST_LEVELS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def distribution_table(pred: pd.DataFrame, calibration: list[dict]) -> pd.DataFrame:
    """Calibration and sharpness of the model's distribution vs the no-model one.

    coverage@q is the share of days that came in at or under the q-quantile;
    a calibrated forecast has coverage@q = q. Mean pinball loss averaged over
    the 19 levels approximates CRPS, in covers: lower means sharper and still honest.
    """
    by_cut = {c["cutoff"]: c for c in calibration}
    rows = {m: {"pinball": [], **{f"coverage@{q:g}": [] for q in DIST_LEVELS}} for m in ("mean4", "model")}
    for cutoff, g in pred.groupby("cutoff"):
        c = by_cut[pd.Timestamp(cutoff).date().isoformat()]
        a = g["actual"].to_numpy()
        qs = {
            "model": dist.quantiles(g["center_log"].to_numpy(), g["scale_log"].to_numpy(), np.array(c["model_z"]), DIST_LEVELS),
            "mean4": dist.quantiles(np.log1p(g["mean4"].to_numpy()), np.ones(len(g)), np.array(c["mean4_z"]), DIST_LEVELS),
        }
        for m, Q in qs.items():
            rows[m]["pinball"].append(np.mean([dist.pinball(Q[:, i], a, q).sum() for i, q in enumerate(DIST_LEVELS)]))
            for i, q in enumerate(DIST_LEVELS):
                rows[m][f"coverage@{q:g}"].append((a <= Q[:, i]).sum())
    n = len(pred)
    out = {}
    for m, r in rows.items():
        out[m] = {"mean_pinball_covers": float(np.sum(r["pinball"]) / n)}
        for q in DIST_LEVELS:
            out[m][f"coverage@{q:g}"] = float(np.sum(r[f"coverage@{q:g}"]) / n)
    return pd.DataFrame(out).T
