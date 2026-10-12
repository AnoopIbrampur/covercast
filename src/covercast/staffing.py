"""What a cover forecast error costs once it becomes a schedule."""

from __future__ import annotations

import numpy as np

from .config import StaffingCosts

# Scalars a baseline forecast is allowed to be multiplied by before staffing.
# A manager who knows understaffing hurts will pad "last week" a bit; giving
# the baselines that same padding, tuned on past data, keeps the comparison fair.
MULTIPLIER_GRID = np.round(np.arange(0.70, 1.81, 0.01), 2)


def hours_over_under(forecast: np.ndarray, actual: np.ndarray, costs: StaffingCosts) -> tuple[np.ndarray, np.ndarray]:
    gap = (np.asarray(forecast, float) - np.asarray(actual, float)) / costs.covers_per_labor_hour
    return np.maximum(gap, 0.0), np.maximum(-gap, 0.0)


def labor_cost(forecast: np.ndarray, actual: np.ndarray, costs: StaffingCosts) -> np.ndarray:
    """Dollars wasted per store-day: idle hours at wage, short hours at wage x under_multiplier."""
    over, under = hours_over_under(forecast, actual, costs)
    return costs.wage * (over + costs.under_multiplier * under)


def best_multiplier(forecast: np.ndarray, actual: np.ndarray, costs: StaffingCosts) -> float:
    f, a = np.asarray(forecast, float), np.asarray(actual, float)
    totals = [labor_cost(k * f, a, costs).sum() for k in MULTIPLIER_GRID]
    return float(MULTIPLIER_GRID[int(np.argmin(totals))])
