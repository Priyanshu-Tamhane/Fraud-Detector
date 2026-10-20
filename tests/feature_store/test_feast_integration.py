"""Tests for Feast Declarative Feature Store: Point-in-time correctness, materialization, and parity."""

import gc
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from feast import Entity, FeatureStore, FeatureView, Field, FileSource, RepoConfig, ValueType
from feast.types import Float32, Int64

from feature_store.dataset_builder import build_device_features, build_user_features


def test_build_feature_tables_structure() -> None:
    """Verifies that batch feature table builders generate valid Feast schemas and UTC timestamps."""
    raw_df = pd.DataFrame(
        {
            "transaction_id": ["tx_1", "tx_2", "tx_3"],
            "user_id": ["u_10", "u_10", "u_20"],
            "device_id": ["dev_1", "dev_2", "dev_1"],
            "amount": [100.0, 300.0, 50.0],
            "timestamp": [
                datetime(2026, 1, 1, 10, 0, 0),
                datetime(2026, 1, 1, 11, 0, 0),
                datetime(2026, 1, 1, 12, 0, 0),
            ],
            "is_fraud": [0, 1, 0],
        }
    )

    user_feats = build_user_features(raw_df)
    device_feats = build_device_features(raw_df)

    # 1. Assert user features schema and types
    assert "user_id" in user_feats.columns
    assert "event_timestamp" in user_feats.columns
    assert "created_timestamp" in user_feats.columns
    assert "user_prior_count" in user_feats.columns
    assert "user_prior_mean_amount" in user_feats.columns
    assert user_feats["event_timestamp"].dt.tz is not None  # UTC tz-aware
    assert user_feats["user_prior_count"].iloc[0] == 0
    assert user_feats["user_prior_count"].iloc[1] == 1
    assert user_feats["user_prior_mean_amount"].iloc[1] == 100.0

    # 2. Assert device features schema and types
    assert "device_id" in device_feats.columns
    assert "event_timestamp" in device_feats.columns
    assert "device_unique_users_seen" in device_feats.columns
    assert device_feats["device_unique_users_seen"].iloc[0] == 1
    # dev_1 seen by u_10 on row 0, then u_20 on row 2 -> 2 unique users
    assert (
        device_feats[device_feats["device_id"] == "dev_1"]["device_unique_users_seen"].iloc[1] == 2
    )


def test_feast_point_in_time_correctness_and_parity(tmp_path: Path) -> None:
    """Tests that Feast get_historical_features guarantees point-in-time correctness with zero leakage,

    and that materialized online features achieve 100% parity with the latest offline state.
    """
    store_dir = tmp_path / "feast_repo"
    store_dir.mkdir(parents=True, exist_ok=True)

    user_pq = store_dir / "user_features.parquet"
    device_pq = store_dir / "device_features.parquet"
    registry_db = store_dir / "registry.db"
    online_db = store_dir / "online_store.db"

    # 1. Create historical batch data
    # User u_alpha has 2 transactions: at 10:00 (prior_count=0) and at 12:00 (prior_count=1, prior_mean=100.0)
    user_batch_df = pd.DataFrame(
        {
            "user_id": ["u_alpha", "u_alpha"],
            "event_timestamp": [
                datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC),
                datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
            ],
            "created_timestamp": [
                datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC),
                datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
            ],
            "user_prior_count": [0, 1],
            "user_prior_mean_amount": [0.0, 100.0],
            "user_prior_sum": [0.0, 100.0],
            "user_fraud_count": [0, 0],
        }
    )
    user_batch_df.to_parquet(user_pq, index=False)

    # Device dev_omega has 1 transaction: at 10:00 (unique_users=1, flagged=0)
    device_batch_df = pd.DataFrame(
        {
            "device_id": ["dev_omega"],
            "event_timestamp": [datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)],
            "created_timestamp": [datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)],
            "device_unique_users_seen": [1],
            "is_device_flagged": [0],
        }
    )
    device_batch_df.to_parquet(device_pq, index=False)

    # 2. Define entities & feature views
    u_entity = Entity(name="user_id", value_type=ValueType.STRING, join_keys=["user_id"])
    d_entity = Entity(name="device_id", value_type=ValueType.STRING, join_keys=["device_id"])

    u_fv = FeatureView(
        name="user_behavior_fv",
        entities=[u_entity],
        ttl=timedelta(days=30),
        schema=[
            Field(name="user_prior_count", dtype=Int64),
            Field(name="user_prior_mean_amount", dtype=Float32),
            Field(name="user_prior_sum", dtype=Float32),
            Field(name="user_fraud_count", dtype=Int64),
        ],
        source=FileSource(
            path=str(user_pq.resolve()),
            timestamp_field="event_timestamp",
            created_timestamp_column="created_timestamp",
        ),
    )

    d_fv = FeatureView(
        name="device_reputation_fv",
        entities=[d_entity],
        ttl=timedelta(days=30),
        schema=[
            Field(name="device_unique_users_seen", dtype=Int64),
            Field(name="is_device_flagged", dtype=Int64),
        ],
        source=FileSource(
            path=str(device_pq.resolve()),
            timestamp_field="event_timestamp",
            created_timestamp_column="created_timestamp",
        ),
    )

    # 3. Initialize Feast FeatureStore with SQLite online store
    repo_cfg = RepoConfig(
        project="fraudshield_test",
        registry=str(registry_db.resolve()),
        provider="local",
        online_store={"type": "sqlite", "path": str(online_db.resolve())},
        entity_key_serialization_version=3,
    )
    fs = FeatureStore(config=repo_cfg)
    fs.apply([u_entity, d_entity, u_fv, d_fv])

    # 4. Test Point-in-Time Correctness (Zero Lookahead Leakage)
    # Query at 10:30: should strictly see state from 10:00 (count=0, mean=0.0)
    # Query at 12:30: should strictly see state from 12:00 (count=1, mean=100.0)
    entity_eval_df = pd.DataFrame(
        {
            "user_id": ["u_alpha", "u_alpha"],
            "device_id": ["dev_omega", "dev_omega"],
            "timestamp": [
                datetime(2026, 1, 1, 10, 30, 0, tzinfo=UTC),
                datetime(2026, 1, 1, 12, 30, 0, tzinfo=UTC),
            ],
        }
    )

    features_to_fetch = [
        "user_behavior_fv:user_prior_count",
        "user_behavior_fv:user_prior_mean_amount",
        "device_reputation_fv:device_unique_users_seen",
    ]

    hist_df = fs.get_historical_features(
        entity_df=entity_eval_df,
        features=features_to_fetch,
    ).to_df()

    # Evaluation row 0: at 10:30 after 1st tx but before 2nd tx
    # Must NOT see the 2nd transaction that occurred at 12:00
    assert hist_df["user_prior_count"].iloc[0] == 0
    assert hist_df["user_prior_mean_amount"].iloc[0] == 0.0
    assert hist_df["device_unique_users_seen"].iloc[0] == 1

    # Evaluation row 1: at 12:30 after 2nd tx
    # Must see updated count and mean
    assert hist_df["user_prior_count"].iloc[1] == 1
    assert hist_df["user_prior_mean_amount"].iloc[1] == 100.0

    # 5. Test Online Materialization
    fs.materialize(
        start_date=datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC),
        end_date=datetime(2026, 1, 1, 23, 59, 0, tzinfo=UTC),
    )

    # 6. Test Online Retrieval & Parity
    online_response = fs.get_online_features(
        features=features_to_fetch,
        entity_rows=[{"user_id": "u_alpha", "device_id": "dev_omega"}],
    ).to_df()

    # Online store must immediately serve the latest state (count=1, mean=100.0, unique_users=1)
    assert online_response["user_prior_count"].iloc[0] == 1
    assert np.isclose(online_response["user_prior_mean_amount"].iloc[0], 100.0)
    assert online_response["device_unique_users_seen"].iloc[0] == 1

    # Mathematical Parity: Online retrieval exactly matches historical feature extraction at T = 12:30
    assert online_response["user_prior_count"].iloc[0] == hist_df["user_prior_count"].iloc[1]
    assert np.isclose(
        online_response["user_prior_mean_amount"].iloc[0],
        hist_df["user_prior_mean_amount"].iloc[1],
    )

    # Explicit cleanup to release SQLite locks on Windows
    del fs
    gc.collect()
    try:
        shutil.rmtree(store_dir, ignore_errors=True)
    except Exception:
        pass


def test_feature_store_client_workflow(tmp_path: Path) -> None:
    """Verifies FeatureStoreClient wrapper methods for applying registry and online lookup."""
    from feature_store.client import FeatureStoreClient

    client_dir = tmp_path / "client_repo"
    client_dir.mkdir(parents=True, exist_ok=True)
    reg_db = client_dir / "reg.db"
    online_db = client_dir / "online.db"

    client = FeatureStoreClient(
        online_store_type="sqlite",
        registry_path=reg_db,
        online_db_path=online_db,
    )

    # 1. Apply feature views
    client.apply()
    assert reg_db.exists()

    # 2. Lookup single online features
    single = client.get_online_features_single(user_id="user_test_99", device_id="device_test_99")
    assert isinstance(single, dict)
    assert "user_id" in single
    assert "device_id" in single

    del client
    gc.collect()
