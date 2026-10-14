"""Thread-safe model artifact loader for real-time inference."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml.config import settings
from ml.logging import logger


class ModelLoader:
    """Manages loading, caching, and serving trained fraud detection model artifacts."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or settings.api.model_path
        self._bundle: dict[str, Any] | None = None
        self._is_loaded: bool = False

    def load(self) -> None:
        """Loads model bundle from disk into memory."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at '{self.model_path}'. "
                "Ensure Phase 1 training pipeline has been executed."
            )

        logger.info(f"Loading fraud model artifact from '{self.model_path}'...")
        bundle = joblib.load(self.model_path)

        # Validate bundle contents
        required_keys = {"model_name", "model", "optimal_threshold", "feature_names"}
        missing_keys = required_keys - set(bundle.keys())
        if missing_keys:
            raise ValueError(f"Invalid model bundle. Missing required keys: {missing_keys}")

        self._bundle = bundle
        self._is_loaded = True
        logger.info(
            f"Model loaded successfully: {self.model_name} | "
            f"Features: {len(self.feature_names)} | Optimal Threshold: {self.optimal_threshold:.4f}"
        )

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def model_name(self) -> str:
        self._check_loaded()
        return str(self._bundle["model_name"])  # type: ignore[index]

    @property
    def feature_names(self) -> list[str]:
        self._check_loaded()
        return list(self._bundle["feature_names"])  # type: ignore[index]

    @property
    def optimal_threshold(self) -> float:
        self._check_loaded()
        return float(self._bundle["optimal_threshold"])  # type: ignore[index]

    @property
    def metrics(self) -> dict[str, Any]:
        self._check_loaded()
        return dict(self._bundle.get("metrics", {}))  # type: ignore[index]

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Calculates calibrated fraud probabilities for feature inputs."""
        self._check_loaded()
        calibrator = self._bundle.get("calibrator")  # type: ignore[index]
        if calibrator is not None and hasattr(calibrator, "predict_proba"):
            return calibrator.predict_proba(X)

        # Fallback to base model if calibrator not present
        model = self._bundle["model"]  # type: ignore[index]
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            return probs[:, 1] if probs.ndim == 2 else probs
        raise ValueError("Loaded model cannot produce predicted probabilities.")

    def _check_loaded(self) -> None:
        if not self._is_loaded or self._bundle is None:
            raise RuntimeError("Model is not loaded. Call 'load()' first.")


# Default global model loader singleton
model_loader = ModelLoader()
