"""Leak-free feature engineering pipeline for tabular and behavioral fraud features."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.logging import logger


@dataclass
class FeatureDataset:
    """Container for engineered features and targets."""

    features: pd.DataFrame
    labels: pd.Series
    feature_names: list[str]


class FeaturePipeline:
    """Constructs tabular, temporal, velocity, and behavioral features without lookahead leakage."""

    def __init__(self) -> None:
        self.category_freq_map: dict[str, float] = {}
        self.fitted: bool = False
        self.feature_columns: list[str] = []

    def fit(self, train_df: pd.DataFrame) -> "FeaturePipeline":
        """Learns encodings strictly from training data."""
        # Learn frequency encoding for merchant categories from train partition
        cat_counts = train_df["merchant_category"].value_counts(normalize=True)
        self.category_freq_map = cat_counts.to_dict()
        self.fitted = True
        return self

    def extract_features(
        self, df: pd.DataFrame, state_df: pd.DataFrame | None = None
    ) -> FeatureDataset:
        """Extracts leakage-free features.

        Args:
            df: The partition to extract features for.
            state_df: Optional cumulative historical context (e.g. train history when processing validation).
        """
        if not self.fitted:
            raise RuntimeError(
                "FeaturePipeline must be fitted on training data before extracting features."
            )

        # Ensure input is sorted chronologically
        working_df = df.copy()
        working_df["timestamp"] = pd.to_datetime(working_df["timestamp"])
        working_df.sort_values(by="timestamp", inplace=True)

        # 1. Base Transaction Features
        features_df = pd.DataFrame(index=working_df.index)
        features_df["amount"] = working_df["amount"].astype(float)
        features_df["log_amount"] = np.log1p(features_df["amount"]) # transactions can be way too large so this prevents the model from being biased

        # Temporal cyclics
        dt = working_df["timestamp"].dt
        features_df["hour_of_day"] = dt.hour
        features_df["day_of_week"] = dt.dayofweek
        features_df["is_weekend"] = dt.dayofweek.isin([5, 6]).astype(int)
        features_df["is_night"] = dt.hour.isin([0, 1, 2, 3, 4, 5]).astype(int)

        # Frequency encode merchant category
        features_df["cat_frequency"] = (
            working_df["merchant_category"]
            .map(lambda c: self.category_freq_map.get(c, 0.001))
            .astype(float)
        )

        # 2. Cumulative / Velocity features strictly looking backward
        # Combine with state_df if provided to maintain temporal memory
        if state_df is not None:
            combined = pd.concat([state_df, working_df], ignore_index=True)
            combined["timestamp"] = pd.to_datetime(combined["timestamp"])
            combined.sort_values(by="timestamp", inplace=True)
            eval_indices = set(working_df["transaction_id"].values)
        else:
            combined = working_df
            eval_indices = set(working_df["transaction_id"].values)

        # Compute rolling & expanding metrics per user
        logger.info(
            f"Computing velocity & behavioral features across {len(working_df):,} records..."
        )

        # Cumulative user stats (shifted by 1 to exclude current transaction from historical average)
        user_grouped = combined.groupby("user_id")

        # User prior transaction count
        user_prior_count = user_grouped.cumcount()

        # User prior cumulative sum and expanding mean
        user_prior_sum = user_grouped["amount"].cumsum() - combined["amount"]
        user_prior_mean = user_prior_sum / np.maximum(user_prior_count, 1)

        combined["user_prior_count"] = user_prior_count
        combined["user_prior_mean_amount"] = user_prior_mean
        combined["amount_to_user_mean_ratio"] = combined["amount"] / np.maximum(
            user_prior_mean, 10.0
        )

        # Device behavioral tracking (is_new_device for this user)
        # Check if pair (user_id, device_id) was seen before
        combined["user_device_pair"] = combined["user_id"] + "::" + combined["device_id"]
        pair_cumcount = combined.groupby("user_device_pair").cumcount()
        combined["is_new_device_for_user"] = (pair_cumcount == 0).astype(int)

        # Device risk: historical users per device
        device_grouped = combined.groupby("device_id")
        combined["device_unique_users_seen"] = device_grouped["user_id"].transform(
            lambda s: (~s.duplicated()).cumsum()
        )

        # Filter back strictly to the target partition
        target_rows = combined[combined["transaction_id"].isin(eval_indices)].copy()
        target_rows.sort_values(by="timestamp", inplace=True)

        # Assign computed features
        features_df["user_prior_count"] = target_rows["user_prior_count"].values
        features_df["user_prior_mean_amount"] = target_rows["user_prior_mean_amount"].values
        features_df["amount_to_user_mean_ratio"] = target_rows["amount_to_user_mean_ratio"].values
        features_df["is_new_device_for_user"] = target_rows["is_new_device_for_user"].values
        features_df["device_unique_users_seen"] = target_rows["device_unique_users_seen"].values

        # Store feature columns list
        self.feature_columns = list(features_df.columns)
        labels = working_df["is_fraud"].copy().reset_index(drop=True)

        return FeatureDataset(
            features=features_df.reset_index(drop=True),
            labels=labels,
            feature_names=self.feature_columns,
        )
