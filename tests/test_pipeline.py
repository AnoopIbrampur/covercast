from __future__ import annotations

import json

import numpy as np

from covercast import report
from covercast.backtest import run


def test_hpg_bookings_are_mapped_and_closed_days_are_nan(synth_dir, synth_panel):
    import pandas as pd

    rel = pd.read_csv(synth_dir / "store_id_relation.csv")
    hpg = pd.read_csv(synth_dir / "hpg_reserve.csv")
    air = pd.read_csv(synth_dir / "air_reserve.csv")
    assert len(rel) and len(hpg)
    assert len(synth_panel.res_visitors) == len(air) + hpg["hpg_store_id"].isin(rel["hpg_store_id"]).sum()
    visits = pd.read_csv(synth_dir / "air_visit_data.csv")
    assert np.isfinite(synth_panel.visitors).sum() == len(visits)


def test_backtest_end_to_end(synth_panel, tmp_path):
    result = run(synth_panel, n_eval=2, verbose=False)
    pred = result.predictions
    assert pred["cutoff"].nunique() == 2
    assert pred["actual"].notna().all()
    for col in ["last_week", "mean4", "model", "model_quantile", "model_padded"]:
        assert (pred[col] >= 0).all()

    summary = report.write(result, tmp_path, "test")
    assert (tmp_path / "backtest.png").exists()
    loaded = json.loads((tmp_path / "summary.json").read_text())
    assert loaded["n_eval_cutoffs"] == 2
    # The quantile model is asked for the 2/3 quantile, so it should come up
    # short on roughly a third of days, and far less often than a mean forecast.
    short = summary["staffing"]
    assert short["model_quantile"]["short_day_rate"] < short["model"]["short_day_rate"]


def test_report_rebuilds_from_saved_predictions(synth_panel, tmp_path):
    result = run(synth_panel, n_eval=2, verbose=False)
    first = report.write(result, tmp_path, "test")
    again = report.rebuild(tmp_path, "test")
    assert again["staffing"] == first["staffing"]
    seg = again["segments"]
    assert seg["all"]["store_days"] == len(result.predictions)
    assert seg["takes bookings"]["store_days"] + seg["no bookings"]["store_days"] == seg["all"]["store_days"]
