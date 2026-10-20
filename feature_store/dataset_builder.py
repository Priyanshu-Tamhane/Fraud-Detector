"""Builds and serializes Parquet batch feature tables for Feast."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ml.config import settings
from ml.data.generator import TransactionGenerator
from ml.logging import logger


def build_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes leak-free historical user features strictly as of each transaction timestamp."""
    working_df = df.copy()
    working_df["timestamp"] = pd.to_datetime(working_df["timestamp"])
    working_df.sort_values(by="timestamp", inplace=True)

    # Ensure timestamps are UTC tz-aware
    if working_df["timestamp"].dt.tz is None:
        working_df["timestamp"] = working_df["timestamp"].dt.tz_localize(UTC)
    else:
        working_df["timestamp"] = working_df["timestamp"].dt.tz_convert(UTC)

    user_grouped = working_df.groupby("user_id")

    # Historical metrics strictly prior to current transaction
    user_prior_count = user_grouped.cumcount()
    user_prior_sum = user_grouped["amount"].cumsum() - working_df["amount"]
    user_prior_mean = user_prior_sum / np.maximum(user_prior_count, 1)

    # Historical confirmed fraud count prior to current transaction
    if "is_fraud" in working_df.columns:
        user_fraud_count = user_grouped["is_fraud"].cumsum() - working_df["is_fraud"]
    else:
        user_fraud_count = pd.Series(0, index=working_df.index)

    user_features_df = pd.DataFrame(
        {
            "user_id": working_df["user_id"].astype(str),
            "event_timestamp": working_df["timestamp"],
            "created_timestamp": datetime.now(tz=UTC),
            "user_prior_count": user_prior_count.astype("int64"),
            "user_prior_mean_amount": user_prior_mean.astype("float32"),
            "user_prior_sum": user_prior_sum.astype("float32"),
            "user_fraud_count": user_fraud_count.astype("int64"),
        }
    )

    return user_features_df.reset_index(drop=True)


def build_device_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes leak-free historical device reputation features as of each transaction."""
    working_df = df.copy()
    working_df["timestamp"] = pd.to_datetime(working_df["timestamp"])
    working_df.sort_values(by="timestamp", inplace=True)

    # Ensure timestamps are UTC tz-aware
    if working_df["timestamp"].dt.tz is None:
        working_df["timestamp"] = working_df["timestamp"].dt.tz_localize(UTC)
    else:
        working_df["timestamp"] = working_df["timestamp"].dt.tz_convert(UTC)

    device_grouped = working_df.groupby("device_id")

    # Cumulative distinct users seen on this device
    device_unique_users = device_grouped["user_id"].transform(lambda s: (~s.duplicated()).cumsum())

    # Whether device was previously associated with confirmed fraud
    if "is_fraud" in working_df.columns:
        device_prior_fraud = device_grouped["is_fraud"].cumsum() - working_df["is_fraud"]
        is_device_flagged = (device_prior_fraud > 0).astype("int64")
    else:
        is_device_flagged = pd.Series(0, index=working_df.index)

    device_features_df = pd.DataFrame(
        {
            "device_id": working_df["device_id"].astype(str),
            "event_timestamp": working_df["timestamp"],
            "created_timestamp": datetime.now(tz=UTC),
            "device_unique_users_seen": device_unique_users.astype("int64"),
            "is_device_flagged": is_device_flagged.astype("int64"),
        }
    )

    return device_features_df.reset_index(drop=True)


def build_and_save_feature_tables(
    transactions_df: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    """Builds and writes user & device Parquet feature tables to disk for Feast offline store."""
    if transactions_df is None:
        raw_path = Path(settings.data.raw_data_path)
        if raw_path.exists():
            logger.info(f"Loading raw transactions from '{raw_path}'...")
            transactions_df = pd.read_parquet(raw_path)
        else:
            logger.info("Generating synthetic transactions for feature store...")
            generator = TransactionGenerator(seed=settings.data.random_seed)
            transactions_df = generator.generate(
                n_samples=settings.data.sample_size,
                fraud_rate=settings.data.fraud_rate,
            )

    logger.info("Extracting user behavior feature table...")
    user_df = build_user_features(transactions_df)

    logger.info("Extracting device reputation feature table...")
    device_df = build_device_features(transactions_df)

    user_out = Path(settings.feast.user_features_parquet)
    device_out = Path(settings.feast.device_features_parquet)

    user_out.parent.mkdir(parents=True, exist_ok=True)
    device_out.parent.mkdir(parents=True, exist_ok=True)

    user_df.to_parquet(user_out, index=False)
    device_df.to_parquet(device_out, index=False)

    logger.info(f"Saved {len(user_df):,} user feature rows to '{user_out}'.")
    logger.info(f"Saved {len(device_df):,} device feature rows to '{device_out}'.")

    return user_out, device_out
