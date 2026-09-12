"""Single-stage event BDT; reconstruction and sample weighting belong to callers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

FEATURES = (
    "M_HJ", "M_LJ", "D2_LJ", "E_HJ", "E_LJ",
    "abs_cos_theta_HJ", "abs_cos_theta_LJ", "D2_HJ",
    "constituent_multiplicity_HJ", "constituent_multiplicity_LJ",
    "track_multiplicity_HJ", "track_multiplicity_LJ",
    "charged_energy_fraction_HJ", "charged_energy_fraction_LJ",
    "jet_girth_HJ", "jet_girth_LJ", "tau21_HJ", "tau21_LJ",
)
HISTORICAL_THRESHOLD = 0.7963983416557312
NUM_BOOST_ROUND = 85
PARAMS = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "eval_metric": "logloss",
    "nthread": 1,
    "seed": 314159,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "scale_pos_weight": 1.0,
    "max_depth": 4,
    "min_child_weight": 5.0,
    "eta": 0.08,
}


class T0R015BDT:
    """Fit the formal recipe on caller-supplied selected reconstructed events.

    X must contain exactly FEATURES in order. Labels (0=background, 1=signal)
    and sample weights are separate, positionally aligned one-dimensional inputs.
    No splitting, reweighting, imputation, scaling, or threshold fitting occurs.
    """

    @staticmethod
    def _matrix(X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame with named features")
        if tuple(X.columns) != FEATURES:
            raise ValueError("X columns must match FEATURES exactly, including order")
        if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
            raise ValueError("Features must have real numeric dtypes")
        if any(pd.api.types.is_complex_dtype(dtype) for dtype in X.dtypes):
            raise ValueError("Features must have real numeric dtypes")
        matrix = X.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
        if np.isinf(matrix).any():
            raise ValueError("Infinite features are invalid; use NaN for missing values")
        return matrix

    @staticmethod
    def _vector(values, X: pd.DataFrame, name: str) -> np.ndarray:
        if isinstance(values, pd.Series) and not values.index.equals(X.index):
            raise ValueError(f"{name} Series index must match X.index exactly")
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (len(X),) or not np.isfinite(result).all():
            raise ValueError(f"{name} must be a finite vector with one value per row")
        return result

    def fit(self, X: pd.DataFrame, y, *, sample_weight) -> "T0R015BDT":
        """Train 85 rounds with explicit weights; see README for historical weights."""
        matrix = self._matrix(X)
        labels = self._vector(y, X, "y")
        weights = self._vector(sample_weight, X, "sample_weight")
        if set(labels) != {0.0, 1.0}:
            raise ValueError("y must contain both binary classes 0 and 1")
        if (weights < 0).any() or any(weights[labels == c].sum() <= 0 for c in (0, 1)):
            raise ValueError("Weights must be nonnegative with positive mass in both classes")
        dtrain = xgb.DMatrix(
            matrix, label=labels.astype(np.int8), weight=weights,
            feature_names=list(FEATURES), missing=np.nan, nthread=1,
        )
        self.booster_ = xgb.train(
            dict(PARAMS), dtrain, num_boost_round=NUM_BOOST_ROUND,
            evals=[(dtrain, "train")], verbose_eval=False,
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Return one signal-class score per row, following the BDT repository API.

        These logistic outputs are not calibrated physical signal probabilities.
        """
        if not hasattr(self, "booster_"):
            raise RuntimeError("Call fit before predict_proba")
        matrix = self._matrix(X)
        if len(X) == 0:
            return pd.Series(index=X.index, dtype=np.float32, name="score")
        data = xgb.DMatrix(matrix, feature_names=list(FEATURES), missing=np.nan, nthread=1)
        return pd.Series(self.booster_.predict(data), index=X.index, name="score")

    def score(self, X: pd.DataFrame) -> pd.Series:
        """Alias for predict_proba; this is an event score, not an accuracy metric."""
        return self.predict_proba(X)
