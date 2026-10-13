from __future__ import annotations

import numpy as np

from covercast.classical import booked_by_lead, pickup, weekday_smoothing

from .conftest import make_panel


def test_smoothing_recovers_a_clean_weekly_pattern():
    T, ci = 140, 120
    pattern = np.array([10, 12, 14, 16, 30, 40, 20], float)
    dates_dow0 = 0  # make_panel starts on Monday 2016-01-04
    V = np.array([[pattern[(dates_dow0 + t) % 7] for t in range(T)]])
    fc = weekday_smoothing(make_panel(V), ci, np.array([0]), alpha=0.3, alpha_dev=0.2, phi=0.9)
    expected = [pattern[(ci + h) % 7] for h in range(1, 15)]
    np.testing.assert_allclose(fc[0], expected, rtol=1e-9)


def test_smoothing_ignores_closed_days_and_never_looks_past_cutoff():
    T, ci = 140, 120
    V = np.full((1, T), 20.0)
    V[0, ::7] = np.nan  # closed every Monday
    V[0, ci + 1 :] = 999.0  # the future must not leak in
    fc = weekday_smoothing(make_panel(V), ci, np.array([0]), alpha=0.3, alpha_dev=0.2, phi=0.9)
    np.testing.assert_allclose(fc[0], 20.0)


def test_bookings_by_lead_counts_only_bookings_made_early_enough():
    V = np.full((1, 40), 20.0)
    bookings = [(0, 30, 30, 2.0), (0, 30, 27, 3.0), (0, 30, 20, 4.0)]  # leads 0, 3, 10
    K = booked_by_lead(make_panel(V, bookings))
    assert K[0, 30, 0] == 9 and K[0, 30, 1] == 7 and K[0, 30, 3] == 7 and K[0, 30, 4] == 4 and K[0, 30, 11] == 0


def test_pickup_adds_the_usual_late_arrivals_to_what_is_booked():
    T, ci = 140, 120
    V = np.full((1, T), 20.0)
    # Every past same-weekday day had 6 covers booked 3+ days ahead, so 14 arrived later.
    target = ci + 3
    bookings = [(0, d, d - 5, 6.0) for d in range(target - 7 * 8, target, 7)]
    panel = make_panel(V, bookings)
    booked_now = np.zeros((1, 14))
    booked_now[0, 2] = 11.0  # 11 already booked for the target, 3 days out
    fc = pickup(panel, booked_by_lead(panel), ci, np.array([0]), booked_now)
    assert fc[0, 2] == 11.0 + 14.0
