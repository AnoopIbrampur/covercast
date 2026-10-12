from __future__ import annotations

import numpy as np

from covercast.config import StaffingCosts
from covercast.staffing import best_multiplier, hours_over_under, labor_cost


def test_cost_arithmetic():
    c = StaffingCosts(wage=20, covers_per_labor_hour=4, under_multiplier=3)
    over, under = hours_over_under(np.array([48, 32]), np.array([40, 40]), c)
    assert np.allclose(over, [2, 0]) and np.allclose(under, [0, 2])
    assert np.allclose(labor_cost(np.array([48, 32]), np.array([40, 40]), c), [40, 120])


def test_critical_ratio():
    assert StaffingCosts(under_multiplier=1).critical_ratio == 0.5
    assert np.isclose(StaffingCosts(under_multiplier=2).critical_ratio, 2 / 3)


def test_padding_moves_toward_newsvendor_quantile():
    # Demand = forecast x lognormal noise. With short hours costing 3x idle
    # ones, the best padding is the 75th percentile of the noise.
    rng = np.random.default_rng(0)
    f = rng.uniform(10, 60, 20000)
    noise = np.exp(rng.normal(0, 0.3, f.size))
    k = best_multiplier(f, f * noise, StaffingCosts(under_multiplier=3))
    assert abs(k - np.quantile(noise, 0.75)) < 0.03
    assert best_multiplier(f, f, StaffingCosts()) == 1.0
