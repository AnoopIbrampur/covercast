from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, hstack, identity, kron

from covercast.schedule import (
    HOURS,
    ScheduleParams,
    arrival_profiles,
    open_hours,
    replay_cost,
    service_hour,
    solve,
    staff_need,
)

from .conftest import make_panel

PARAMS = ScheduleParams()


def _scenario_milp(need: np.ndarray, open_mask: np.ndarray, p: ScheduleParams) -> float:
    """The textbook formulation, one shortfall variable per scenario-hour. Slow, but obviously right."""
    A = p.shifts
    K, H = need.shape
    J = A.shape[0]
    c = np.concatenate([p.wage * A.sum(1), np.full(K * H, p.under_multiplier * p.wage / K)])
    cover = csr_matrix(A.T)
    cons = [
        LinearConstraint(hstack([kron(np.ones((K, 1)), cover), identity(K * H)]), lb=need.ravel(), ub=np.inf),
        LinearConstraint(hstack([cover[open_mask], csr_matrix((int(open_mask.sum()), K * H))]), lb=p.min_crew, ub=np.inf),
    ]
    res = milp(c, constraints=cons, integrality=np.r_[np.ones(J), np.zeros(K * H)], bounds=Bounds(0, np.inf))
    return res.fun


def _objective(floor: np.ndarray, need: np.ndarray, p: ScheduleParams) -> float:
    return p.wage * floor.sum() + p.under_multiplier * p.wage * np.maximum(need - floor, 0).mean(0).sum()


def _profile() -> np.ndarray:
    prof = np.zeros(len(HOURS))
    prof[3:13] = [0.03, 0.04, 0.03, 0.02, 0.02, 0.1, 0.28, 0.25, 0.15, 0.08]
    return prof / prof.sum()


@pytest.mark.parametrize("min_crew", [1, 2])
def test_fast_solver_matches_scenario_formulation(min_crew):
    p = ScheduleParams(min_crew=min_crew)
    rng = np.random.default_rng(min_crew)
    prof = _profile()
    om = open_hours(prof, p)
    for _ in range(10):
        totals = np.round(rng.uniform(5, 150) * np.exp(rng.normal(0, 0.4, 25))).astype(int)
        need = staff_need(rng.multinomial(totals, prof).astype(float), p)
        floor = solve(need, om, p)
        assert _objective(floor, need, p) == pytest.approx(_scenario_milp(need, om, p), abs=1e-6)


def test_minimum_crew_on_open_hours_and_nothing_else_when_quiet():
    prof = _profile()
    om = open_hours(prof, PARAMS)
    floor = solve(np.zeros((1, len(HOURS))), om, PARAMS)
    assert (floor[om] >= 1).all()
    # The cheapest way to cover the open block with 3-8h shifts never schedules two people at once.
    assert floor.max() == 1
    assert floor.sum() == pytest.approx(max(om.sum(), 3))


def test_busy_night_gets_more_people_at_the_peak():
    prof = _profile()
    floor = solve(staff_need(120 * prof, PARAMS)[None], open_hours(prof, PARAMS), PARAMS)
    peak = int(np.argmax(prof))
    assert floor[peak] > floor[open_hours(prof, PARAMS)].min()


def test_staff_need_spreads_each_cover_over_two_hours():
    arr = np.zeros(len(HOURS))
    arr[5] = 8
    need = staff_need(arr, PARAMS)  # 8 covers x 15 min = 2 staff-hours
    assert need.sum() == pytest.approx(2.0)
    assert need[5] == need[6] == pytest.approx(1.0)


def test_service_hour_wraps_past_midnight():
    assert service_hour(np.array([9, 18, 0, 2, 3, 5])).tolist() == [0, 9, 15, 17, -1, -1]


def test_replay_cost_arithmetic():
    floor = np.array([1, 2, 2, 0], float)
    need = np.array([0.5, 3.0, 1.0, 1.0])
    cost, hours, short = replay_cost(floor, need, ScheduleParams(wage=20, under_multiplier=3))
    assert (hours, short) == (5.0, 2.0)
    assert cost == pytest.approx(20 * 5 + 3 * 20 * 2)


def test_profiles_use_only_bookings_up_to_cutoff_and_shrink_to_genre():
    T, ci = 140, 120
    V = np.full((3, T), 20.0)
    bookings = [(0, d, d - 1, 2.0, 12) for d in range(30, 120)]  # store 0: lunch place, many bookings
    bookings += [(1, 100, 99, 2.0, 20)]  # store 1: a single dinner booking
    bookings += [(0, ci + 3, ci - 2, 50.0, 22)]  # future visit: must be ignored
    prof = arrival_profiles(make_panel(V, bookings), ci, PARAMS)
    lunch, late = 12 - HOURS[0], 22 - HOURS[0]
    assert prof.arrival[0, lunch] > 0.6 and prof.arrival[0, late] < 0.05
    # One booking barely moves store 1 off its genre (which store 0 dominates).
    assert prof.arrival[1, lunch] > 0.5
    # No bookings at all: exactly the genre profile.
    np.testing.assert_allclose(prof.arrival[2], prof.genre[0] / prof.genre[0].sum())
    np.testing.assert_allclose(prof.arrival.sum(axis=1), 1.0)
