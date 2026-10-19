"""Feature Parity Verification Suite: Ensures online features match offline batch features identically."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from ml.features.online.redis_store import OnlineFeatureStore
from ml.features.pipeline import FeaturePipeline


def test_online_offline_feature_parity() -> None:
    """Tests that online features computed sequentially match offline batch features identically (0% skew)."""
    start_time = datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)

    # Deterministic sequence of transactions across users and devices
    raw_records = [
        # User 1: 3 transactions with different amounts and a new device on tx 3
        {
            "transaction_id": "tx_p_01",
            "user_id": "user_p1",
            "merchant_id": "merch_01",
            "device_id": "dev_01",
            "amount": 100.0,
            "timestamp": start_time,
            "merchant_category": "retail",
            "is_fraud": 0,
        },
        {
            "transaction_id": "tx_p_02",
            "user_id": "user_p1",
            "merchant_id": "merch_02",
            "device_id": "dev_01",  # Known device
            "amount": 250.0,
            "timestamp": start_time + timedelta(minutes=2),  # within 5m
            "merchant_category": "grocery",
            "is_fraud": 0,
        },
        {
            "transaction_id": "tx_p_03",
            "user_id": "user_p1",
            "merchant_id": "merch_01",
            "device_id": "dev_02",  # Brand new device for user_p1
            "amount": 800.0,
            "timestamp": start_time + timedelta(minutes=10),  # after 5m, within 1h
            "merchant_category": "retail",
            "is_fraud": 0,
        },
        # User 2: Uses dev_01 (shared device with user_p1)
        {
            "transaction_id": "tx_p_04",
            "user_id": "user_p2",
            "merchant_id": "merch_03",
            "device_id": "dev_01",  # Device now shared across 2 users
            "amount": 50.0,
            "timestamp": start_time + timedelta(minutes=15),
            "merchant_category": "dining",
            "is_fraud": 0,
        },
    ]

    df = pd.DataFrame(raw_records)

    # 1. Compute Offline Batch Features
    offline_pipe = FeaturePipeline()
    offline_pipe.fit(df)
    offline_dataset = offline_pipe.extract_features(df)
    offline_df = offline_dataset.features

    # 2. Compute Online Real-Time Features Sequentially
    online_store = OnlineFeatureStore()
    online_store.clear()

    online_records: list[dict] = []
    for row in raw_records:
        feats = online_store.get_and_update_features_sync(
            user_id=row["user_id"],
            device_id=row["device_id"],
            amount=row["amount"],
            timestamp=row["timestamp"],
            tx_id=row["transaction_id"],
            update_state=True,
        )
        online_records.append(
            {
                "user_prior_count": feats.user_prior_count,
                "user_prior_mean_amount": feats.user_prior_mean_amount,
                "amount_to_user_mean_ratio": feats.amount_to_user_mean_ratio,
                "is_new_device_for_user": feats.is_new_device_for_user,
                "device_unique_users_seen": feats.device_unique_users_seen,
            }
        )

    online_df = pd.DataFrame(online_records)

    # 3. Assert Mathematical Parity Across All Records (0% Skew)
    for col in [
        "user_prior_count",
        "user_prior_mean_amount",
        "amount_to_user_mean_ratio",
        "is_new_device_for_user",
        "device_unique_users_seen",
    ]:
        offline_vals = offline_df[col].values
        online_vals = online_df[col].values

        if np.issubdtype(offline_vals.dtype, np.integer):
            np.testing.assert_array_equal(
                online_vals,
                offline_vals,
                err_msg=f"Discrepancy detected in integer feature '{col}'!",
            )
        else:
            np.testing.assert_allclose(
                online_vals,
                offline_vals,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Discrepancy detected in floating point feature '{col}'!",
            )


def test_sliding_velocity_window_counts() -> None:
    """Verifies that sliding velocity windows accurately count transactions across 5m, 1h, and 24h."""
    store = OnlineFeatureStore()
    store.clear()
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

    # Tx 1 at 12:00
    f1 = store.get_and_update_features_sync("u_vel", "dev_01", 100.0, t0, "tx1")
    assert f1.tx_count_5m == 0
    assert f1.tx_count_1h == 0
    assert f1.tx_count_24h == 0

    # Tx 2 at 12:02 (within 5 min)
    f2 = store.get_and_update_features_sync(
        "u_vel", "dev_01", 50.0, t0 + timedelta(minutes=2), "tx2"
    )
    assert f2.tx_count_5m == 1
    assert f2.tx_count_1h == 1
    assert f2.tx_count_24h == 1

    # Tx 3 at 12:04 (still within 5 min)
    f3 = store.get_and_update_features_sync(
        "u_vel", "dev_01", 75.0, t0 + timedelta(minutes=4), "tx3"
    )
    assert f3.tx_count_5m == 2
    assert f3.tx_count_1h == 2
    assert f3.tx_count_24h == 2

    # Tx 4 at 12:20 (after 5 min, but within 1 hour)
    f4 = store.get_and_update_features_sync(
        "u_vel", "dev_01", 200.0, t0 + timedelta(minutes=20), "tx4"
    )
    assert f4.tx_count_5m == 0  # previous transactions were > 5m ago
    assert f4.tx_count_1h == 3  # all 3 are within 1h
    assert f4.tx_count_24h == 3
    assert f4.amount_sum_24h == 225.0  # 100 + 50 + 75
