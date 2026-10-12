"""Write a backtest's tables, a summary JSON and one figure to a results folder."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .backtest import (  # noqa: E402
    BacktestResult,
    accuracy_table,
    by_horizon,
    per_cutoff_wins,
    segment_table,
    staffing_table,
    store_bootstrap,
)


def write(result: BacktestResult, out_dir: Path, label: str) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred, costs = result.predictions, result.costs

    acc = accuracy_table(pred)
    staff = staffing_table(pred, costs)
    hz = by_horizon(pred)
    seg = segment_table(pred, costs)
    best_base = acc.loc[["last_week", "mean4"], "rmsle"].idxmin()
    best_staff_base = staff.loc[["last_week_padded", "mean4_padded"], "cost_per_store_week"].idxmin()

    summary = {
        "label": label,
        "n_eval_cutoffs": int(pred["cutoff"].nunique()),
        "eval_window": [str(pred["target_date"].min().date()), str(pred["target_date"].max().date())],
        "n_stores": int(pred["store"].nunique()),
        "n_forecasts": int(len(pred)),
        "costs": asdict(costs) | {"critical_ratio": costs.critical_ratio},
        "accuracy": acc.round(4).to_dict(orient="index"),
        "staffing": staff.round(3).to_dict(orient="index"),
        "model_vs_best_baseline_rmsle": {
            "baseline": best_base,
            **store_bootstrap(pred, "model", best_base),
            "cutoff_wins": per_cutoff_wins(pred, "model", best_base),
        },
        "quantile_vs_best_padded_baseline_cost": {
            "baseline": best_staff_base,
            **store_bootstrap(pred, "model_quantile", best_staff_base, costs),
            "cutoff_wins": per_cutoff_wins(pred, "model_quantile", best_staff_base, costs),
        },
        "segments": seg.round(4).to_dict(orient="index"),
        "padding": result.multipliers,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    pred.to_csv(out_dir / "predictions.csv.gz", index=False)

    md = [
        f"# Backtest: {label}",
        "",
        f"{summary['n_eval_cutoffs']} weekly cutoffs, target days {summary['eval_window'][0]} to "
        f"{summary['eval_window'][1]}, {summary['n_stores']} stores, {summary['n_forecasts']:,} store-day forecasts.",
        "",
        "## Accuracy",
        "",
        acc.round(4).to_markdown(),
        "",
        "## Staffing cost",
        "",
        f"Wage ${costs.wage:.0f}/h, {costs.covers_per_labor_hour:g} covers per variable labor hour, "
        f"short hours cost {costs.under_multiplier:g}x. Cost-minimising quantile = {costs.critical_ratio:.3f}.",
        "",
        staff.round(2).to_markdown(),
        "",
        "## Where it holds",
        "",
        "Accuracy vs the average of the last four same weekdays, and labor cost (per store-week) vs that",
        "baseline with its best padding.",
        "",
        seg.round(4).to_markdown(),
        "",
        "## RMSLE by horizon (days ahead)",
        "",
        hz.round(4).to_markdown(),
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(md))
    _figure(hz, staff, out_dir / "backtest.png", label)
    return summary


def rebuild(results_dir: Path, label: str) -> dict:
    """Re-write the report from saved predictions, without retraining anything."""
    import pandas as pd

    from .config import StaffingCosts

    results_dir = Path(results_dir)
    old = json.loads((results_dir / "summary.json").read_text())
    pred = pd.read_csv(results_dir / "predictions.csv.gz", parse_dates=["cutoff", "target_date"])
    c = old["costs"]
    costs = StaffingCosts(c["wage"], c["covers_per_labor_hour"], c["under_multiplier"])
    return write(BacktestResult(pred, costs, old.get("padding", [])), results_dir, label)


def _figure(hz, staff, path: Path, label: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    names = {"last_week": "Same day last week", "mean4": "Avg of last 4 same weekdays", "model": "Model"}
    for m, name in names.items():
        ax1.plot(hz.index, hz[m], marker="o", ms=3, label=name)
    ax1.set_xlabel("Days ahead")
    ax1.set_ylabel("RMSLE (lower is better)")
    ax1.set_title("Forecast error by lead time")
    ax1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8)

    order = ["mean4", "model", "mean4_padded", "model_padded", "model_quantile"]
    labels = ["Avg of 4\nas-is", "Model\nas-is", "Avg of 4\npadded", "Model\npadded", "Model\nquantile"]
    colors = ["#9aa5b1", "#8fb3f5", "#9aa5b1", "#5b8def", "#1f5fd1"]
    vals = staff.loc[order, "cost_per_store_week"]
    ax2.bar(labels, vals, color=colors)
    for i, v in enumerate(vals):
        ax2.text(i, v, f"${v:,.0f}", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("Mis-staffing cost per store-week ($)")
    ax2.set_title("What the error costs in labor")
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(label, fontsize=10, color="#555")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
