from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from covercast.data import Panel, load_panel
from covercast.synthetic import generate


def make_panel(visitors: np.ndarray, bookings: list[tuple[int, int, int, float]] = (), start="2016-01-04") -> Panel:
    """Hand-built panel. bookings are (store, visit_day, booked_day, covers)."""
    S, T = visitors.shape
    dates = pd.date_range(start, periods=T, freq="D")
    b = np.array(bookings, dtype=float).reshape(-1, 4)
    return Panel(
        store_ids=np.array([f"s{i}" for i in range(S)]),
        dates=dates,
        visitors=visitors.astype(float),
        genre=np.zeros(S, int),
        pref=np.zeros(S, int),
        genre_names=["g"],
        pref_names=["p"],
        holiday=pd.Series(0, index=pd.date_range(dates[0] - pd.Timedelta(days=1), periods=T + 31)),
        res_store=b[:, 0].astype(int),
        res_visit_day=b[:, 1].astype(int),
        res_booked_day=b[:, 2].astype(int),
        res_visitors=b[:, 3],
    )


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory):
    return generate(tmp_path_factory.mktemp("synth"), n_stores=25, seed=3)


@pytest.fixture(scope="session")
def synth_panel(synth_dir):
    return load_panel(synth_dir)
