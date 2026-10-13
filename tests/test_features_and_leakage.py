"""Unit tests for temporal splitting and zero-leakage feature pipeline."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from ml.features.pipeline import FeaturePipeline
from ml.features.temporal_split import TemporalSplitter


def test_temporal_splitter_guarantees_chronology() -> None:
    start = datetime(2026, 1, 1, 0, 0, 0)
    data = [
        {
            "transaction_id": f"tx_{i}",
            "timestamp": start + timedelta(minutes=i),
            "is_fraud": 0,
        }
        for i in range(100)
    ]
    df = pd.DataFrame(data)
    splitter = TemporalSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    splits = splitter.split(df)

    assert len(splits.train_df) == 70
    assert len(splits.val_df) == 15
    assert len(splits.test_df) == 15

    assert splits.train_df["timestamp"].max() <= splits.val_df["timestamp"].min()
    assert splits.val_df["timestamp"].max() <= splits.test_df["timestamp"].min()


def test_temporal_splitter_rejects_invalid_ratios() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        TemporalSplitter(train_ratio=0.5, val_ratio=0.2, test_ratio=0.2)


def test_feature_pipeline_no_lookahead_leakage() -> None:
    start = datetime(2026, 1, 1, 10, 0, 0)
    # User u_1 makes 3 transactions at 10:00, 10:30, 11:00
    df = pd.DataFrame(
        [
            {
                "transaction_id": "tx_1",
                "user_id": "u_1",
                "merchant_id": "m_1",
                "device_id": "d_1",
                "amount": 100.0,
                "timestamp": start,
                "merchant_category": "retail",
                "is_fraud": 0,
            },
            {
                "transaction_id": "tx_2",
                "user_id": "u_1",
                "merchant_id": "m_1",
                "device_id": "d_1",
                "amount": 200.0,
                "timestamp": start + timedelta(minutes=30),
                "merchant_category": "retail",
                "is_fraud": 0,
            },
            {
                "transaction_id": "tx_3",
                "user_id": "u_1",
                "merchant_id": "m_1",
                "device_id": "d_2",
                "amount": 300.0,
                "timestamp": start + timedelta(minutes=60),
                "merchant_category": "retail",
                "is_fraud": 0,
            },
        ]
    )

    pipeline = FeaturePipeline()
    pipeline.fit(df)
    dataset = pipeline.extract_features(df)
    feats = dataset.features

    # For tx_1: first transaction, prior count must be 0, prior mean must be 0
    assert feats.loc[0, "user_prior_count"] == 0
    assert feats.loc[0, "user_prior_mean_amount"] == 0.0
    assert feats.loc[0, "is_new_device_for_user"] == 1

    # For tx_2: second transaction, prior count must be 1, prior mean must be 100.0
    assert feats.loc[1, "user_prior_count"] == 1
    assert feats.loc[1, "user_prior_mean_amount"] == 100.0
    # device d_1 was seen before, so not new
    assert feats.loc[1, "is_new_device_for_user"] == 0

    # For tx_3: third transaction, prior count must be 2, prior mean must be (100+200)/2 = 150.0
    assert feats.loc[2, "user_prior_count"] == 2
    assert feats.loc[2, "user_prior_mean_amount"] == 150.0
    # device d_2 is new for user u_1
    assert feats.loc[2, "is_new_device_for_user"] == 1
