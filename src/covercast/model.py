"""One global gradient-boosted model across all stores and horizons.

The target is log1p(covers). Squared error on that scale is RMSLE, the
competition metric, and it stops a 300-cover Saturday from drowning out
every 15-cover Tuesday. Quantile regression also works on the log scale,
because a quantile survives any increasing transform: the q-th quantile of
log1p(covers) maps back to the q-th quantile of covers.

The trees learn each day relative to the store's own level (its average for
this weekday over the last 8 weeks, in log terms) rather than the raw level.
Trees can't extrapolate, and with mostly small restaurants in training an
absolute-target model pulled the busiest stores down toward the middle: on the
60+ covers/day stores it under-forecast by 31%, and the shift backtest
understaffed them badly. Predicting the ratio left every other size band
unchanged and cut that bias to 13%. The offset is a function of the features,
so quantiles of the ratio are still quantiles of covers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import CATEGORICAL, FEATURES

PARAMS = dict(
    learning_rate=0.05,
    max_iter=400,
    max_leaf_nodes=31,
    min_samples_leaf=50,
    l2_regularization=1.0,
    early_stopping=False,
    random_state=0,
)


def level(df: pd.DataFrame) -> np.ndarray:
    """The store's own log level for the target weekday, known at the cutoff."""
    return df["sd_mean8"].fillna(df["mean_28"]).fillna(0.0).to_numpy(dtype=float)


def _xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    return df[FEATURES].astype(float), np.log1p(df["actual"].to_numpy(dtype=float)) - level(df)


def fit(train: pd.DataFrame, quantile: float | None = None) -> HistGradientBoostingRegressor:
    X, y = _xy(train)
    loss = dict(loss="squared_error") if quantile is None else dict(loss="quantile", quantile=quantile)
    model = HistGradientBoostingRegressor(categorical_features=CATEGORICAL, **loss, **PARAMS)
    return model.fit(X, y)


def predict_log(model: HistGradientBoostingRegressor, df: pd.DataFrame) -> np.ndarray:
    return model.predict(df[FEATURES].astype(float)) + level(df)


def predict(model: HistGradientBoostingRegressor, df: pd.DataFrame) -> np.ndarray:
    return np.clip(np.expm1(predict_log(model, df)), 0, None)
