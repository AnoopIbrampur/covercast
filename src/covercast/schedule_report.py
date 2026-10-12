"""Tables and a figure for the shift-level backtest."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .schedule import ScheduleParams  # noqa: E402
from .schedule_eval import POLICIES, by_volume, params_dict, summary_table  # noqa: E402

LABELS = {
    "mean4_padded": "4-week avg\n+ padding",
    "model_padded": "Model\n+ padding",
    "mean4_dist": "4-week avg\nrange, optimised",
    "model_dist": "Model distribution,\noptimised",
}


def store_bootstrap_excess(sched: pd.DataFrame, a: str, b: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """(a - b) excess cost per store-week, store-level block bootstrap."""
    g = sched.assign(d=sched[f"cost_{a}"] - sched[f"cost_{b}"]).groupby("store").agg(d=("d", "sum"), n=("d", "size"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    draws = g["d"].to_numpy()[idx].sum(1) / (g["n"].to_numpy()[idx].sum(1) / 7)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    point = g["d"].sum() / (g["n"].sum() / 7)
    wins = sum(
        (c[f"cost_{a}"].sum() < c[f"cost_{b}"].sum()) for _, c in sched.groupby("cutoff")
    )
    return {"diff": float(point), "ci_low": float(lo), "ci_high": float(hi),
            "cutoff_wins": [int(wins), int(sched["cutoff"].nunique())]}


def write(sched: pd.DataFrame, params: ScheduleParams, out_dir: Path, label: str) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tab = summary_table(sched, params)
    vol = by_volume(sched)
    cmp = store_bootstrap_excess(sched, "model_dist", "mean4_padded")
    cmp_pad = store_bootstrap_excess(sched, "model_dist", "model_padded")
    summary = {
        "label": label,
        "params": params_dict(params),
        "store_days": len(sched),
        "stores": int(sched["store"].nunique()),
        "per_store_week": tab.round(3).to_dict(orient="index"),
        "by_volume": vol.round(4).to_dict(orient="index"),
        "model_dist_vs_mean4_padded": cmp,
        "model_dist_vs_model_padded": cmp_pad,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    sched.to_csv(out_dir / "schedules.csv.gz", index=False)

    md = [
        f"# Shift-level backtest: {label}",
        "",
        f"{len(sched):,} store-days, {sched['store'].nunique()} stores. Wage ${params.wage:g}/h, "
        f"{params.labor_minutes_per_cover:g} staff-minutes per cover, short staff-hours cost "
        f"{params.under_multiplier:g}x, minimum crew {params.min_crew}, shifts of "
        f"{min(params.shift_lengths)}-{max(params.shift_lengths)}h, {params.n_scenarios} demand scenarios.",
        "",
        "## Per store-week",
        "",
        tab.round(2).to_markdown(),
        "",
        "Policies: " + "; ".join(f"`{k}` = {v}" for k, v in POLICIES.items()) + ".",
        "",
        "## Excess over perfect foresight, by store volume ($ per store-week)",
        "",
        vol.round(2).to_markdown(),
        "",
        f"Model distribution vs padded 4-week average: {cmp['diff']:+.2f} $/store-week "
        f"[{cmp['ci_low']:+.2f}, {cmp['ci_high']:+.2f}], {cmp['cutoff_wins'][0]}/{cmp['cutoff_wins'][1]} weeks.",
        "",
        f"Model distribution vs padded model mean: {cmp_pad['diff']:+.2f} $/store-week "
        f"[{cmp_pad['ci_low']:+.2f}, {cmp_pad['ci_high']:+.2f}], {cmp_pad['cutoff_wins'][0]}/{cmp_pad['cutoff_wins'][1]} weeks.",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(md))
    _figure(tab, vol, out_dir / "schedule.png", label)
    return summary


def _figure(tab: pd.DataFrame, vol: pd.DataFrame, path: Path, label: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    order = list(LABELS)
    colors = ["#9aa5b1", "#8fb3f5", "#c2c9d1", "#1f5fd1"]
    vals = tab.loc[order, "excess_over_oracle"]
    ax1.bar([LABELS[o] for o in order], vals, color=colors)
    for i, v in enumerate(vals):
        ax1.text(i, v, f"${v:,.0f}", ha="center", va="bottom", fontsize=9)
    ax1.set_ylabel("Cost above perfect-foresight schedule\n($ per store-week)")
    ax1.set_title("Real shifts, replayed against the real day")
    ax1.tick_params(axis="x", labelsize=8)

    x = np.arange(len(vol))
    w = 0.38
    ax2.bar(x - w / 2, vol["mean4_padded"], w, label="4-week avg + padding", color="#9aa5b1")
    ax2.bar(x + w / 2, vol["model_dist"], w, label="Model distribution, optimised", color="#1f5fd1")
    ax2.set_xticks(x, [f"{b}\n({n / 7:,.0f} store-weeks)" for b, n in zip(vol.index, vol["store_days"])], fontsize=8)
    ax2.set_ylabel("$ above perfect foresight per store-week")
    ax2.set_title("Where forecasting pays: by store volume")
    ax2.legend(frameon=False, fontsize=8)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(label, fontsize=10, color="#555")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
