"""Build the data behind the demo dashboard (site/index.html reads site/data.json).

One cutoff, a dozen stores across volume bands, and for each of their next 14
days: the forecast distribution, the schedule each policy would post, the
shifts themselves, what actually happened, and why the model moved.

The "why" is counterfactual, not a feature-importance score: re-predict with
one input set to its ordinary value (bookings at this store's usual level for
this lead time, no holiday, the last week running at the 4-week pace) and
report how far the forecast moves. Effects are each measured holding the
others fixed, so they don't add up exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import distribution as dist
from . import model as mdl
from .backtest import training_cutoffs
from .config import HORIZON, StaffingCosts
from .data import Panel
from .features import build_cutoff, cutoff_indices
from .schedule import (
    HOURS,
    ScheduleParams,
    arrival_profiles,
    describe_shifts,
    open_hours,
    replay_cost,
    solve_shifts,
    staff_need,
)

BAND_PICKS = [(0, 15, 2), (15, 30, 3), (30, 60, 4), (60, np.inf, 3)]
SHOW_Q = {"p10": 0.1, "p25": 0.25, "p50": 0.5, "p75": 0.75, "p90": 0.9}


def _reasons(point, row: pd.DataFrame, avg_party: float) -> list[dict]:
    base = float(np.expm1(mdl.predict_log(point, row))[0])
    r = row.iloc[0]
    out = []

    known, typical = float(np.expm1(r["res_known"])), float(np.expm1(r["res_typical_at_lead"]))
    if known > 0 or typical > 0:
        cf = row.copy()
        cf["res_known"] = r["res_typical_at_lead"]
        cf["res_parties_known"] = typical / avg_party
        out.append(
            {
                "label": "Bookings so far",
                "detail": f"{known:.0f} covers booked, vs {typical:.1f} usual at {int(r['h'])} day{'s' if r['h'] > 1 else ''} out",
                "effect": base - float(np.expm1(mdl.predict_log(point, cf))[0]),
            }
        )
    if r["holiday"] or r["holiday_prev"] or r["holiday_next"]:
        cf = row.copy()
        cf[["holiday", "holiday_prev", "holiday_next"]] = 0
        what = "Public holiday" if r["holiday"] else ("Day after a holiday" if r["holiday_prev"] else "Eve of a holiday")
        out.append({"label": what, "detail": "", "effect": base - float(np.expm1(mdl.predict_log(point, cf))[0])})
    cf = row.copy()
    cf["mean_7"] = r["mean_28"]
    pace = float(np.expm1(r["mean_7"]) / max(np.expm1(r["mean_28"]), 1e-9) - 1)
    out.append(
        {
            "label": "Recent pace",
            "detail": f"last 7 open days ran {pace:+.0%} vs the 4-week level",
            "effect": base - float(np.expm1(mdl.predict_log(point, cf))[0]),
        }
    )
    for o in out:
        o["effect"] = round(o["effect"], 1)
    return sorted(out, key=lambda o: -abs(o["effect"]))


def build(panel: Panel, results_dir: Path, out_dir: Path, label: str, seed: int = 0) -> Path:
    """Dashboard data for the last evaluation cutoff of a finished backtest.

    Forecasts, spreads, calibration and padding come from the backtest's saved
    outputs, so the board shows exactly what was scored. The point model is
    refit once at that cutoff only to run the "why" counterfactuals.
    """
    results_dir = Path(results_dir)
    summary = json.loads((results_dir / "summary.json").read_text())
    calibration = json.loads((results_dir / "calibration.json").read_text())
    c = summary["costs"]
    costs = StaffingCosts(c["wage"], c["covers_per_labor_hour"], c["under_multiplier"])
    params = ScheduleParams(wage=costs.wage, labor_minutes_per_cover=60 / costs.covers_per_labor_hour,
                            under_multiplier=costs.under_multiplier)
    pred = pd.read_csv(results_dir / "predictions.csv.gz", parse_dates=["cutoff", "target_date"])
    last = pred["cutoff"].max()
    ci = panel.dates.get_loc(last)
    z_model = np.array(calibration[-1]["model_z"])
    k_mean4 = summary["padding"][-1]["mean4"]

    all_ci, _ = cutoff_indices(panel, n_eval=1)
    train = pd.concat([build_cutoff(panel, t) for t in training_cutoffs(all_ci, ci)], ignore_index=True)
    point = mdl.fit(train[train["actual"].notna()])
    # All of the cutoff's rows (closed days too, which the board shows as gaps),
    # with the backtest's forecasts merged in where the day was scored.
    test = build_cutoff(panel, ci)
    scored = pred[pred["cutoff"] == last][["store", "h", "center_log", "scale_log"]]
    test = test.merge(scored, on=["store", "h"], how="left")
    missing = test["center_log"].isna()
    if missing.any():
        test.loc[missing, "center_log"] = mdl.predict_log(point, test[missing])
        test.loc[missing, "scale_log"] = np.nan
    prof = arrival_profiles(panel, ci, params)
    avg_party = float(np.mean(panel.res_visitors)) if len(panel.res_visitors) else 2.5

    level = test.groupby("store")["mean_28"].first().pipe(np.expm1)
    rng = np.random.default_rng(seed)
    picks = []
    for lo, hi, n in BAND_PICKS:
        pool = level[(level >= lo) & (level < hi)].index.to_numpy()
        picks += list(rng.choice(pool, min(n, len(pool)), replace=False))

    # Closed days have no scored spread; give them the store's median spread so the band still draws.
    test["scale_log"] = test.groupby("store")["scale_log"].transform(lambda v: v.fillna(v.median())).fillna(0.4)
    levels = (np.arange(params.n_scenarios) + 0.5) / params.n_scenarios
    stores = []
    for s in sorted(picks, key=lambda s: level[s]):
        g = test[test["store"] == s].sort_values("h")
        p_s, om = prof.arrival[s], open_hours(prof.arrival[s], params)
        days = []
        for _, r in g.iterrows():
            row = g[g["h"] == r["h"]]
            q = dist.quantiles(np.array([r["center_log"]]), np.array([r["scale_log"]]), z_model,
                               np.array(list(SHOW_Q.values()) + [costs.critical_ratio]))[0]
            scen_rng = np.random.default_rng(int(s) * 1000 + int(r["h"]))
            totals = dist.quantiles(np.array([r["center_log"]]), np.array([r["scale_log"]]), z_model, levels)[0]
            scen_need = staff_need(scen_rng.multinomial(np.round(totals).astype(int), p_s).astype(float), params)
            x_model = solve_shifts(scen_need, om, params)
            base_need = staff_need(k_mean4 * r["mean4"] * p_s, params)[None]
            x_base = solve_shifts(base_need, om, params)
            floor_m, floor_b = x_model @ params.shifts, x_base @ params.shifts

            day = {
                "date": r["target_date"].date().isoformat(),
                "dow": r["target_date"].strftime("%a"),
                "h": int(r["h"]),
                "holiday": int(r["holiday"]),
                "q": {k: round(float(v), 1) for k, v in zip(SHOW_Q, q[:-1])},
                "staff_to": round(float(q[-1]), 1),
                "mean4_padded": round(float(k_mean4 * r["mean4"]), 1),
                "reasons": _reasons(point, row, avg_party),
                "need": {
                    "p10": np.quantile(scen_need, 0.1, axis=0).round(2).tolist(),
                    "p50": np.quantile(scen_need, 0.5, axis=0).round(2).tolist(),
                    "p90": np.quantile(scen_need, 0.9, axis=0).round(2).tolist(),
                },
                "schedule": {
                    "model": {"floor": floor_m.astype(int).tolist(), "shifts": describe_shifts(x_model, params),
                              "hours": float(floor_m.sum())},
                    "baseline": {"floor": floor_b.astype(int).tolist(), "shifts": describe_shifts(x_base, params),
                                 "hours": float(floor_b.sum())},
                },
                "actual": None,
            }
            if not np.isnan(r["actual"]):
                real = np.random.default_rng(int(s) * 7 + int(r["h"])).multinomial(int(r["actual"]), p_s).astype(float)
                need_real = staff_need(real, params)
                floor_o = solve_shifts(need_real[None], om, params) @ params.shifts
                day["actual"] = int(r["actual"])
                day["need_real"] = need_real.round(2).tolist()
                day["replay"] = {
                    name: round(replay_cost(f, need_real, params)[0], 2)
                    for name, f in (("model", floor_m), ("baseline", floor_b), ("oracle", floor_o))
                }
            days.append(day)
        stores.append(
            {
                "id": f"Store {int(s):03d}",
                "genre": panel.genre_names[panel.genre[s]],
                "area": panel.pref_names[panel.pref[s]],
                "typical": round(float(level[s]), 1),
                "bookings_seen": int(prof.n_bookings[s]),
                "open_hours": [int(HOURS[om].min()), int(HOURS[om].max()) + 1] if om.any() else None,
                "days": days,
            }
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "label": label,
            "cutoff": panel.dates[ci].date().isoformat(),
            "horizon": HORIZON,
            "hours": HOURS.tolist(),
            "wage": params.wage,
            "under_multiplier": params.under_multiplier,
            "critical_ratio": costs.critical_ratio,
            "min_crew": params.min_crew,
            "labor_minutes_per_cover": params.labor_minutes_per_cover,
            "baseline_padding": k_mean4,
        },
        "stores": stores,
    }
    path = out_dir / "data.json"
    path.write_text(json.dumps(payload, separators=(",", ":")))
    render(path, out_dir)
    return path


TEMPLATE = Path(__file__).with_name("site_template.html")


def render(data_path: Path, out_dir: Path) -> None:
    """index.html is a complete page for GitHub Pages; body.html is the same page without the document shell."""
    data = Path(data_path).read_text().replace("</", "<\\/")  # keep "</script>" out of the inline JSON
    body = TEMPLATE.read_text().replace("__DATA__", data)
    (Path(out_dir) / "body.html").write_text(body)
    shell = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
    )
    head, rest = body.split("<style>", 1)
    page = shell + head + "<style>" + rest.replace("</style>", "</style>\n</head>\n<body>", 1) + "\n</body>\n</html>\n"
    (Path(out_dir) / "index.html").write_text(page)
