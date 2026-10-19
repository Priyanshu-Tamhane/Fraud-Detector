"""Real-time transaction feature transformation and fraud score prediction."""

import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from ml.config import settings
from ml.features.online.redis_store import OnlineFeatures, OnlineFeatureStore, online_feature_store
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
    """Computes online features via Redis/feature store, scores transactions, and explains risk factors."""

    def __init__(
        self,
        loader: ModelLoader | None = None,
        decider: DecisionEngine | None = None,
        store: OnlineFeatureStore | None = None,
    ) -> None:
        self.loader = loader or model_loader
        self.decider = decider or decision_engine
        self.store = store or online_feature_store

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

        # 1. Compute backward-looking behavioral & velocity features from online store
        online_feats: OnlineFeatures = self.store.get_and_update_features_sync(
            user_id=user_id,
            device_id=device_id,
            amount=amount,
            timestamp=timestamp,
            tx_id=transaction_id,
            update_state=update_state,
        )

        # 2. Instantaneous Temporal features
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        is_weekend = 1 if day_of_week in (5, 6) else 0
        is_night = 1 if hour in (0, 1, 2, 3, 4, 5) else 0
        cat_freq = self._category_freq.get(merchant_category.lower(), 0.01)

        # Construct single-row feature dictionary matching model's expected schema
        feature_dict = {
            "amount": float(amount),
            "log_amount": float(np.log1p(amount)),
            "hour_of_day": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "is_night": is_night,
            "cat_frequency": cat_freq,
            "user_prior_count": online_feats.user_prior_count,
            "user_prior_mean_amount": online_feats.user_prior_mean_amount,
            "amount_to_user_mean_ratio": online_feats.amount_to_user_mean_ratio,
            "is_new_device_for_user": online_feats.is_new_device_for_user,
            "device_unique_users_seen": online_feats.device_unique_users_seen,
        }

        # Align with model's expected feature order
        feature_df = pd.DataFrame([feature_dict])[self.loader.feature_names]

        # 3. Run inference
        raw_prob = self.loader.predict_proba(feature_df)[0]
        prob = float(np.clip(raw_prob, 0.0, 1.0))

        # 4. Decision
        decision_res: DecisionResult = self.decider.decide(prob)

        # 5. Identify top risk factors (including velocity anomalies)
        risk_factors: list[str] = []
        if online_feats.is_new_device_for_user == 1:
            risk_factors.append("new_device")
        if online_feats.amount_to_user_mean_ratio >= 3.0 and online_feats.user_prior_count > 0:
            risk_factors.append("high_amount_deviation")
        if is_night == 1:
            risk_factors.append("unusual_transaction_time")
        if online_feats.device_unique_users_seen > 1:
            risk_factors.append("device_connected_to_multiple_accounts")
        if online_feats.tx_count_5m >= 4:
            risk_factors.append("high_velocity_burst_5m")
        if amount >= 50_000.0:
            risk_factors.append("large_monetary_amount")
        if merchant_category.lower() in ("crypto_exchange", "electronics") and prob >= 0.05:
            risk_factors.append("high_risk_merchant_category")

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
