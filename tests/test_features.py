from __future__ import annotations

import numpy as np
import pandas as pd

from covercast.backtest import training_cutoffs
from covercast.config import HORIZON
from covercast.features import BASELINES, FEATURES, build_cutoff, cutoff_indices

from .conftest import make_panel


def _scramble_future(panel, ci, seed=0):
    """Same panel, but every visit after ci and every booking made after ci is replaced with noise."""
    rng = np.random.default_rng(seed)
    V = panel.visitors.copy()
    V[:, ci + 1 :] = rng.integers(1, 500, V[:, ci + 1 :].shape)
    late = panel.res_booked_day > ci
    rv = panel.res_visitors.copy()
    rv[late] = rng.integers(1, 50, late.sum())
    vd = panel.res_visit_day.copy()
    vd[late] = ci + rng.integers(1, HORIZON + 1, late.sum())
    return type(panel)(**{**panel.__dict__, "visitors": V, "res_visitors": rv, "res_visit_day": vd})


def test_features_ignore_everything_after_cutoff(synth_panel):
    _, eval_ci = cutoff_indices(synth_panel, n_eval=3)
    for ci in eval_ci:
        clean = build_cutoff(synth_panel, ci)
        dirty = build_cutoff(_scramble_future(synth_panel, ci), ci)
        cols = FEATURES + BASELINES
        pd.testing.assert_frame_equal(clean[cols], dirty[cols])
        # The target is the one column that is supposed to come from the future.
        assert not np.allclose(clean["actual"].fillna(-1), dirty["actual"].fillna(-1))


def test_booking_counts_only_if_made_by_cutoff():
    T, ci = 140, 120
    V = np.full((1, T), 20.0)
    bookings = [
        (0, ci + 3, ci, 4.0),  # made on the cutoff day: known
        (0, ci + 3, ci - 10, 2.0),  # made well before: known
        (0, ci + 3, ci + 1, 8.0),  # made the day after: not known yet
        (0, ci + 9, ci - 1, 5.0),
    ]
    df = build_cutoff(make_panel(V, bookings), ci).set_index("h")
    assert np.isclose(np.expm1(df.loc[3, "res_known"]), 6.0)
    assert df.loc[3, "res_parties_known"] == 2
    assert np.isclose(np.expm1(df.loc[9, "res_known"]), 5.0)
    assert df.loc[1, "res_known"] == 0


def test_typical_bookings_respect_lead_time():
    T, ci = 140, 120
    V = np.full((1, T), 20.0)
    # A past day with one booking made 2 days ahead and one made 10 days ahead.
    bookings = [(0, ci - 5, ci - 7, 28.0), (0, ci - 5, ci - 15, 56.0)]
    df = build_cutoff(make_panel(V, bookings), ci).set_index("h")
    # At lead 1-2 both bookings existed; at lead 3-10 only the early one.
    assert np.isclose(np.expm1(df.loc[2, "res_typical_at_lead"]), 84 / 28)
    assert np.isclose(np.expm1(df.loc[5, "res_typical_at_lead"]), 56 / 28)
    assert np.isclose(np.expm1(df.loc[11, "res_typical_at_lead"]), 0)


def test_last_week_baseline_uses_most_recent_same_weekday():
    T, ci = 140, 120
    V = np.tile(np.arange(T, dtype=float) + 1, (1, 1))  # value = day index + 1
    df = build_cutoff(make_panel(V), ci).set_index("h")
    # h <= 7: one week back. h in 8..14: two weeks back, the latest that is known.
    assert df.loc[1, "last_week"] == (ci + 1 - 7) + 1
    assert df.loc[7, "last_week"] == (ci + 7 - 7) + 1
    assert df.loc[8, "last_week"] == (ci + 8 - 14) + 1
    assert df.loc[14, "last_week"] == (ci + 14 - 14) + 1
    t = ci + 1
    assert np.isclose(df.loc[1, "mean4"], np.mean([t - 7 * j + 1 for j in range(1, 5)]))


def test_last_week_skips_a_closed_day():
    T, ci = 140, 120
    V = np.full((1, T), 30.0)
    V[0, ci + 1 - 7] = np.nan  # last week's same day was closed
    V[0, ci + 1 - 14] = 10.0
    df = build_cutoff(make_panel(V), ci).set_index("h")
    assert df.loc[1, "last_week"] == 10.0


def test_closed_days_are_not_targets():
    T, ci = 140, 120
    V = np.full((1, T), 30.0)
    V[0, ci + 2] = np.nan
    df = build_cutoff(make_panel(V), ci).set_index("h")
    assert np.isnan(df.loc[2, "actual"]) and df.loc[3, "actual"] == 30.0


def test_inactive_store_is_skipped():
    T, ci = 140, 120
    V = np.full((2, T), 30.0)
    V[1, : ci - 5] = np.nan  # opened 6 days before the cutoff
    df = build_cutoff(make_panel(V), ci)
    assert set(df["store"]) == {0}


def test_cutoffs_end_on_last_day_and_training_never_overlaps(synth_panel):
    all_ci, eval_ci = cutoff_indices(synth_panel, n_eval=4)
    assert eval_ci[-1] + HORIZON == synth_panel.n_days - 1
    assert all(b - a == 7 for a, b in zip(all_ci, all_ci[1:]))
    for ci in eval_ci:
        tc = training_cutoffs(all_ci, ci)
        assert tc and max(tc) + HORIZON <= ci
