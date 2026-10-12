from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import RAW_DIR, RESULTS_DIR, SYNTH_DIR, StaffingCosts


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


if __name__ == "__main__":
    main()
