"""One global gradient-boosted model across all stores and horizons.

The target is log1p(covers). Squared error on that scale is RMSLE, the
competition metric, and it stops a 300-cover Saturday from drowning out
every 15-cover Tuesday. Quantile regression also works on the log scale,
because a quantile survives any increasing transform: the q-th quantile of
log1p(covers) maps back to the q-th quantile of covers.
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


def _xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    return df[FEATURES].astype(float), np.log1p(df["actual"].to_numpy(dtype=float))


def fit(train: pd.DataFrame, quantile: float | None = None) -> HistGradientBoostingRegressor:
    X, y = _xy(train)
    loss = dict(loss="squared_error") if quantile is None else dict(loss="quantile", quantile=quantile)
    model = HistGradientBoostingRegressor(categorical_features=CATEGORICAL, **loss, **PARAMS)
    return model.fit(X, y)


def predict(model: HistGradientBoostingRegressor, df: pd.DataFrame) -> np.ndarray:
    return np.clip(np.expm1(model.predict(df[FEATURES].astype(float))), 0, None)
