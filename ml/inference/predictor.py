"""Real-time transaction feature transformation and fraud score prediction."""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from ml.config import settings
from ml.inference.decision_engine import Decision, DecisionEngine, DecisionResult, decision_engine
from ml.inference.model_loader import ModelLoader, model_loader


@dataclass
class ScoredTransaction:
    """Full outcome of a scored transaction."""

    transaction_id: str
    fraud_probability: float
    risk_score: float
    decision: Decision
    model_version: str
    latency_ms: float
    top_risk_factors: list[str] = field(default_factory=list)


class TransactionPredictor:
    """Computes online features, scores transactions, and explains risk factors in real time."""

    def __init__(
        self,
        loader: ModelLoader | None = None,
        decider: DecisionEngine | None = None,
    ) -> None:
        self.loader = loader or model_loader
        self.decider = decider or decision_engine

        # Lightweight in-memory state tracking for online velocity (will integrate with Redis in Phase 4)
        self._user_tx_counts: dict[str, int] = defaultdict(int)
        self._user_tx_sums: dict[str, float] = defaultdict(float)
        self._user_devices: dict[str, set[str]] = defaultdict(set)
        self._device_users: dict[str, set[str]] = defaultdict(set)

        # Baseline category frequency map fallback
        self._category_freq: dict[str, float] = {
            "retail": 0.30,
            "grocery": 0.25,
            "electronics": 0.15,
            "dining": 0.10,
            "entertainment": 0.10,
            "travel": 0.08,
            "crypto_exchange": 0.02,
        }

    def predict(
        self,
        transaction_id: str,
        user_id: str,
        merchant_id: str,
        device_id: str,
        amount: float,
        timestamp: datetime,
        merchant_category: str,
        update_state: bool = True,
    ) -> ScoredTransaction:
        """Transforms transaction into features, scores with calibrated model, and generates explanation."""
        start_time = time.perf_counter()

        # 1. Compute backward-looking features using state before this transaction
        prior_count = self._user_tx_counts[user_id]
        prior_sum = self._user_tx_sums[user_id]
        prior_mean = prior_sum / max(prior_count, 1) if prior_count > 0 else 0.0

        amount_ratio = amount / max(prior_mean, 10.0) if prior_count > 0 else 1.0
        is_new_device = 1 if device_id not in self._user_devices[user_id] else 0
        device_users_count = len(self._device_users[device_id])

        # Temporal features
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        is_weekend = 1 if day_of_week in (5, 6) else 0
        is_night = 1 if hour in (0, 1, 2, 3, 4, 5) else 0
        cat_freq = self._category_freq.get(merchant_category.lower(), 0.01)

        # Construct single-row feature dictionary
        feature_dict = {
            "amount": float(amount),
            "log_amount": float(np.log1p(amount)),
            "hour_of_day": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "is_night": is_night,
            "cat_frequency": cat_freq,
            "user_prior_count": prior_count,
            "user_prior_mean_amount": prior_mean,
            "amount_to_user_mean_ratio": amount_ratio,
            "is_new_device_for_user": is_new_device,
            "device_unique_users_seen": max(device_users_count, 1),
        }

        # Align with model's expected feature order
        feature_df = pd.DataFrame([feature_dict])[self.loader.feature_names]

        # 2. Run inference
        raw_prob = self.loader.predict_proba(feature_df)[0]
        prob = float(np.clip(raw_prob, 0.0, 1.0))

        # 3. Decision
        decision_res: DecisionResult = self.decider.decide(prob)

        # 4. Identify top risk factors
        risk_factors: list[str] = []
        if is_new_device == 1:
            risk_factors.append("new_device")
        if amount_ratio >= 3.0 and prior_count > 0:
            risk_factors.append("high_amount_deviation")
        if is_night == 1:
            risk_factors.append("unusual_transaction_time")
        if device_users_count > 1:
            risk_factors.append("device_connected_to_multiple_accounts")
        if amount >= 50_000.0:
            risk_factors.append("large_monetary_amount")
        if merchant_category.lower() in ("crypto_exchange", "electronics") and prob >= 0.05:
            risk_factors.append("high_risk_merchant_category")

        # 5. Optionally update state for subsequent transactions
        if update_state:
            self._user_tx_counts[user_id] += 1
            self._user_tx_sums[user_id] += amount
            self._user_devices[user_id].add(device_id)
            self._device_users[device_id].add(user_id)

        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        return ScoredTransaction(
            transaction_id=transaction_id,
            fraud_probability=decision_res.fraud_probability,
            risk_score=decision_res.risk_score,
            decision=decision_res.decision,
            model_version=settings.api.model_version,
            latency_ms=latency_ms,
            top_risk_factors=risk_factors,
        )


# Global predictor instance
predictor = TransactionPredictor()
