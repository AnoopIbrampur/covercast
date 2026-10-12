"""A full, calibrated predictive distribution for every store-day.

A single quantile is enough to staff one day with one cost ratio. Building
real shifts needs more than that: how likely a slow night is, how fat the busy
tail is, and what happens at any cost ratio. So covers are modelled as

    log1p(covers) = center + scale * z

where `center` is the point model, `scale` is a second model that predicts how
wrong the point model tends to be for this kind of store-day (a volatile
new izakaya two weeks out vs a steady cafe tomorrow), and z is drawn from the
empirical distribution of standardized errors on recent cutoffs that neither
model was trained on. That last step is split conformal calibration: whatever
shape the errors really have, the quantiles are read off them, not assumed.

Calibration uses the eight most recent cutoffs the model could have been
scored on (their whole horizon is over by the evaluation cutoff). Each of
those cutoffs is forecast by its own walk-forward model, trained exactly as
far behind it as the real forecast is behind the evaluation cutoff:

    ... | A: fit the scale model (4 cutoffs) | B: collect standardized errors (4) | gap | eval
          each row forecast by the model trained up to its own cutoff - 14 days

Matching freshness matters. Calibrating on one older model, trained 4-8 weeks
before the rows it is checked on, carried that model's drift into the
intervals: on Recruit, where stores were growing, the medians came out 5
points too high.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import CATEGORICAL, FEATURES

# z is stored as its quantile function on this grid, which is all that's needed
# downstream and keeps a cutoff's calibration to 199 numbers.
Z_LEVELS = np.linspace(0.005, 0.995, 199)
MIN_SCALE = 0.05

SCALE_PARAMS = dict(
    learning_rate=0.05,
    max_iter=200,
    max_leaf_nodes=15,
    min_samples_leaf=200,
    l2_regularization=1.0,
    early_stopping=False,
    random_state=0,
)


@dataclass
class Calibration:
    scale_model: HistGradientBoostingRegressor
    z_quantiles: np.ndarray  # Q_z at Z_LEVELS
    holdout_forecasts: pd.DataFrame  # walk-forward forecast and actual on the holdout rows, for padding
    n_calibration_rows: int

    def scale(self, df: pd.DataFrame) -> np.ndarray:
        return np.maximum(self.scale_model.predict(df[FEATURES].astype(float)), MIN_SCALE)

    def z(self, q: np.ndarray | float) -> np.ndarray:
        return np.interp(q, Z_LEVELS, self.z_quantiles)


def split_holdout(hold: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Older half of the holdout cutoffs (A) and newer half (B)."""
    cutoffs = np.sort(hold["cutoff"].unique())
    b_start = cutoffs[len(cutoffs) // 2]
    return hold[hold["cutoff"] < b_start], hold[hold["cutoff"] >= b_start]


def calibrate(hold: pd.DataFrame) -> Calibration:
    """hold: rows from the holdout cutoffs with features, `actual` and the walk-forward `pred_log`."""
    a, b = split_holdout(hold)
    resid_a = np.log1p(a["actual"].to_numpy()) - a["pred_log"].to_numpy()
    scale_model = HistGradientBoostingRegressor(categorical_features=CATEGORICAL, **SCALE_PARAMS)
    scale_model.fit(a[FEATURES].astype(float), np.abs(resid_a))

    padding_rows = pd.DataFrame({"forecast": np.clip(np.expm1(hold["pred_log"].to_numpy()), 0, None),
                                 "actual": hold["actual"].to_numpy()})
    cal = Calibration(scale_model, Z_LEVELS * 0, padding_rows, len(b))
    resid_b = np.log1p(b["actual"].to_numpy()) - b["pred_log"].to_numpy()
    cal.z_quantiles = np.quantile(resid_b / cal.scale(b), Z_LEVELS)
    return cal


def baseline_z_quantiles(hold: pd.DataFrame, column: str = "mean4") -> np.ndarray:
    """A no-model distribution: the baseline times its own typical error, read off the same holdout B.

    Every store-day gets the same spread in log terms. That is what a careful
    manager could do with a spreadsheet, and what the scale model has to beat.
    """
    b = split_holdout(hold)[1]
    resid = np.log1p(b["actual"].to_numpy()) - np.log1p(b[column].to_numpy())
    return np.quantile(resid, Z_LEVELS)


def quantiles(center_log: np.ndarray, scale_log: np.ndarray, cal_z: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Covers at quantile levels q for each row: shape (rows, len(q))."""
    zq = np.interp(np.atleast_1d(q), Z_LEVELS, cal_z)
    return np.clip(np.expm1(center_log[:, None] + scale_log[:, None] * zq[None, :]), 0, None)


def pinball(forecast: np.ndarray, actual: np.ndarray, q: float) -> np.ndarray:
    d = actual - forecast
    return np.maximum(q * d, (q - 1) * d)
