"""Backtest and staffing assumptions, kept in one place so they are easy to argue with."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "artifacts" / "raw"
SYNTH_DIR = ROOT / "artifacts" / "synthetic"
RESULTS_DIR = ROOT / "artifacts" / "results"

# Schedules are usually posted one to two weeks out, so every forecast covers
# the 14 days after its cutoff.
HORIZON = 14

# A store must have been open this many days in the 8 weeks before a cutoff to
# be forecast at all. Brand-new stores are a cold-start problem, not the one
# being measured here.
MIN_OPEN_DAYS = 14
ACTIVITY_WINDOW = 56

# Longest lookback any feature uses. Cutoffs earlier than this many days after
# the start of the data are skipped so every feature is fully populated.
MAX_LOOKBACK = 91


@dataclass(frozen=True)
class StaffingCosts:
    """Turns a cover forecast error into dollars.

    Only the part of labor that scales with covers matters, because fixed
    minimum staffing is the same whatever the forecast says. Staffing for F
    covers when A show up leaves (F - A) / covers_per_labor_hour hours idle,
    or (A - F) / covers_per_labor_hour hours short.

    Being short costs more than being idle (overtime, call-ins, slower service
    and lost tables), hence under_multiplier > 1. These defaults are
    assumptions, not measurements; the backtest reports a sensitivity sweep.
    """

    wage: float = 18.0
    covers_per_labor_hour: float = 4.0
    under_multiplier: float = 2.0

    @property
    def critical_ratio(self) -> float:
        """Newsvendor optimum: staff to this quantile of demand to minimise expected cost."""
        return self.under_multiplier / (self.under_multiplier + 1.0)
