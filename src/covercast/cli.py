from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import RAW_DIR, RESULTS_DIR, ROOT, SYNTH_DIR, StaffingCosts


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="covercast")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="write a fake restaurant group in the Recruit CSV schema")
    s.add_argument("--out", type=Path, default=SYNTH_DIR)
    s.add_argument("--stores", type=int, default=120)
    s.add_argument("--seed", type=int, default=0)

    b = sub.add_parser("backtest", help="walk-forward backtest and report")
    b.add_argument("--data-dir", type=Path, default=RAW_DIR)
    b.add_argument("--out", type=Path, default=None)
    b.add_argument("--n-eval", type=int, default=8, help="number of weekly evaluation cutoffs")
    b.add_argument("--wage", type=float, default=StaffingCosts.wage)
    b.add_argument("--covers-per-hour", type=float, default=StaffingCosts.covers_per_labor_hour)
    b.add_argument("--under-multiplier", type=float, default=StaffingCosts.under_multiplier)

    r = sub.add_parser("report", help="rebuild a report from saved predictions, without retraining")
    r.add_argument("results_dir", type=Path)
    r.add_argument("--label", default=None)

    sc = sub.add_parser("schedule", help="shift-level backtest on a finished backtest's forecasts")
    sc.add_argument("results_dir", type=Path)
    sc.add_argument("--data-dir", type=Path, default=RAW_DIR)
    sc.add_argument("--stores", type=int, default=None, help="random subset of stores (default: all)")
    sc.add_argument("--min-crew", type=int, default=1)
    sc.add_argument("--workers", type=int, default=None)

    st = sub.add_parser("site", help="build the dashboard from a finished backtest")
    st.add_argument("results_dir", type=Path)
    st.add_argument("--data-dir", type=Path, default=SYNTH_DIR)
    st.add_argument("--out", type=Path, default=None)

    args = ap.parse_args(argv)
    if args.cmd == "synth":
        from .synthetic import generate

        print(f"wrote {generate(args.out, n_stores=args.stores, seed=args.seed)}")
    elif args.cmd == "backtest":
        from . import report
        from .backtest import run
        from .data import load_panel

        panel = load_panel(args.data_dir)
        print(f"{panel.n_stores} stores, {panel.n_days} days, {len(panel.res_visitors):,} bookings")
        costs = StaffingCosts(args.wage, args.covers_per_hour, args.under_multiplier)
        result = run(panel, n_eval=args.n_eval, costs=costs)
        label = "synthetic data" if "synthetic" in str(args.data_dir) else "Recruit (Kaggle) data"
        out = args.out or RESULTS_DIR / ("synthetic" if "synthetic" in str(args.data_dir) else "recruit")
        summary = report.write(result, out, label)
        print((out / "summary.md").read_text())
        q = summary["quantile_vs_best_padded_baseline_cost"]
        print(f"quantile model vs {q['baseline']}: {q['diff']:+.2f} $/store-week "
              f"[{q['ci_low']:+.2f}, {q['ci_high']:+.2f}], wins {q['cutoff_wins'][0]}/{q['cutoff_wins'][1]} cutoffs")
    elif args.cmd == "report":
        from . import report

        label = args.label or json.loads((args.results_dir / "summary.json").read_text())["label"]
        report.rebuild(args.results_dir, label)
        print((args.results_dir / "summary.md").read_text())
    elif args.cmd == "schedule":
        import pandas as pd

        from . import schedule_report
        from .data import load_panel
        from .schedule import ScheduleParams
        from .schedule_eval import run as run_schedules

        old = json.loads((args.results_dir / "summary.json").read_text())
        c = old["costs"]
        params = ScheduleParams(
            wage=c["wage"],
            labor_minutes_per_cover=60.0 / c["covers_per_labor_hour"],
            under_multiplier=c["under_multiplier"],
            min_crew=args.min_crew,
        )
        pred = pd.read_csv(args.results_dir / "predictions.csv.gz", parse_dates=["cutoff", "target_date"])
        calibration = json.loads((args.results_dir / "calibration.json").read_text())
        panel = load_panel(args.data_dir)
        sched = run_schedules(panel, pred, calibration, params, n_stores=args.stores, workers=args.workers)
        out = args.results_dir / ("schedule" if args.min_crew == 1 else f"schedule_crew{args.min_crew}")
        schedule_report.write(sched, params, out, old["label"])
        print((out / "summary.md").read_text())
    elif args.cmd == "site":
        from .data import load_panel
        from .site import build

        synthetic = "synthetic" in str(args.data_dir)
        label = "synthetic data" if synthetic else "Recruit (Kaggle) data"
        # Recruit's per-store counts can't be republished, so only the synthetic
        # build goes to the public site folder; a real-data build stays local.
        out = args.out or (ROOT / "site" if synthetic else RESULTS_DIR / "site_recruit")
        print(f"wrote {build(load_panel(args.data_dir), args.results_dir, out, label)}")


if __name__ == "__main__":
    main()
