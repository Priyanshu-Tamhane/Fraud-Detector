"""Probability calibration module implementing Platt scaling and Isotonic regression."""

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ModelCalibrator:
    """Calibrates classifier probabilities using Platt scaling or Isotonic regression."""

    def __init__(self, method: Literal["sigmoid", "isotonic"] = "sigmoid") -> None:
        
        self.method = method
        self.base_estimator: Any = None
        self.calibrator: LogisticRegression | IsotonicRegression | None = None

    def fit_calibrator(
        self,
        base_estimator: Any,
        X_val: pd.DataFrame | np.ndarray,
        y_val: pd.Series | np.ndarray,
    ) -> "ModelCalibrator":
        """Fits calibration curve using frozen base estimator on validation partition."""
        self.base_estimator = base_estimator
        val_probs = self._get_raw_probs(base_estimator, X_val)
        y_val_arr = y_val.to_numpy() if isinstance(y_val, pd.Series) else y_val

        if self.method == "sigmoid":
            # Platt scaling: 1D Logistic Regression on raw scores
            self.calibrator = LogisticRegression(C=1.0, max_iter=500)
            self.calibrator.fit(val_probs.reshape(-1, 1), y_val_arr)
        else:
            # Isotonic Regression: non-parametric isotonic fit
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.calibrator.fit(val_probs, y_val_arr)

        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Outputs calibrated fraud probabilities."""
        if self.base_estimator is None or self.calibrator is None:
            raise RuntimeError("Calibrator is not fitted.")

        raw_probs = self._get_raw_probs(self.base_estimator, X)
        if self.method == "sigmoid":
            return self.calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
        else:
            return np.clip(self.calibrator.predict(raw_probs), 0.0, 1.0)

    @staticmethod
    def _get_raw_probs(estimator: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Extracts raw positive-class probabilities from an estimator."""
        if hasattr(estimator, "predict_proba"):
            probs = estimator.predict_proba(X)
            if probs.ndim == 2:
                return probs[:, 1]
            return probs
        elif hasattr(estimator, "decision_function"):
            raw = estimator.decision_function(X)
            return 1.0 / (1.0 + np.exp(-raw))
        else:
            raise ValueError("Estimator must have predict_proba or decision_function.")

    @staticmethod
    def get_calibration_curve(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculates empirical fraction of positives vs mean predicted probabilities."""
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
        return prob_true, prob_pred
