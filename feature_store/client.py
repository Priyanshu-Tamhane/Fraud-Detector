"""High-level Feast Feature Store client for offline training and real-time inference."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from feast import FeatureStore, RepoConfig

from feature_store.features import all_entities, all_feature_views
from ml.config import settings
from ml.logging import logger

DEFAULT_FEAST_FEATURES = [
    "user_behavior_fv:user_prior_count",
    "user_behavior_fv:user_prior_mean_amount",
    "user_behavior_fv:user_prior_sum",
    "user_behavior_fv:user_fraud_count",
    "device_reputation_fv:device_unique_users_seen",
    "device_reputation_fv:is_device_flagged",
]


class FeatureStoreClient:
    """Production wrapper around Feast FeatureStore for FraudShield."""

    def __init__(
        self,
        repo_path: Path | str | None = None,
        online_store_type: str | None = None,
        registry_path: Path | str | None = None,
        online_db_path: Path | str | None = None,
    ) -> None:
        self.repo_path = Path(repo_path or settings.feast.repo_path)
        self.online_store_type = online_store_type or settings.feast.online_store_type

        reg_path = str(Path(registry_path or settings.feast.registry_path).resolve())
        Path(reg_path).parent.mkdir(parents=True, exist_ok=True)

        if self.online_store_type == "redis":
            redis_host_port = settings.redis.redis_url.replace("redis://", "").split("/")[0]
            online_config: dict[str, Any] = {
                "type": "redis",
                "connection_string": redis_host_port,
            }
        else:
            db_path = str(Path(online_db_path or settings.feast.online_store_path).resolve())
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            online_config = {
                "type": "sqlite",
                "path": db_path,
            }

        self.repo_config = RepoConfig(
            project="fraudshield",
            registry=reg_path,
            provider="local",
            online_store=online_config,
            entity_key_serialization_version=3,
        )

        self._store = FeatureStore(config=self.repo_config)
        logger.info(
            f"Initialized Feast FeatureStore (online_store='{self.online_store_type}', project='{self._store.project}')."
        )

    @property
    def store(self) -> FeatureStore:
        """Returns the underlying Feast FeatureStore instance."""
        return self._store

    def apply(self) -> None:
        """Registers entities and feature views into the Feast registry."""
        logger.info("Applying Feast schema (entities and feature views) to registry...")
        self._store.apply(all_entities + all_feature_views)
        logger.info("Feast schema registered successfully.")

    def materialize(self, start_date: datetime, end_date: datetime) -> None:
        """Loads offline batch features into the online store between start_date and end_date."""
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=UTC)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=UTC)

        logger.info(
            f"Materializing features into '{self.online_store_type}' from {start_date.isoformat()} to {end_date.isoformat()}..."
        )
        self._store.materialize(start_date=start_date, end_date=end_date)
        logger.info("Feast materialization complete.")

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        features: list[str] | None = None,
    ) -> pd.DataFrame:
        """Retrieves point-in-time correct features for historical training data without lookahead leakage.

        Args:
            entity_df: DataFrame containing join keys (user_id, device_id) and timestamps.
            features: List of feature references ('feature_view:feature_name').
        """
        feature_list = features or DEFAULT_FEAST_FEATURES
        query_df = entity_df.copy()

        # Feast requires an event timestamp column named 'timestamp' or 'event_timestamp'
        if "timestamp" in query_df.columns:
            query_df["timestamp"] = pd.to_datetime(query_df["timestamp"])
            if query_df["timestamp"].dt.tz is None:
                query_df["timestamp"] = query_df["timestamp"].dt.tz_localize(UTC)
            else:
                query_df["timestamp"] = query_df["timestamp"].dt.tz_convert(UTC)
        elif "event_timestamp" in query_df.columns:
            query_df["event_timestamp"] = pd.to_datetime(query_df["event_timestamp"])
            if query_df["event_timestamp"].dt.tz is None:
                query_df["event_timestamp"] = query_df["event_timestamp"].dt.tz_localize(UTC)
            else:
                query_df["event_timestamp"] = query_df["event_timestamp"].dt.tz_convert(UTC)
            query_df.rename(columns={"event_timestamp": "timestamp"}, inplace=True)
        else:
            raise ValueError(
                "entity_df must contain a 'timestamp' or 'event_timestamp' column for point-in-time joins."
            )

        # Ensure join keys are strings
        if "user_id" in query_df.columns:
            query_df["user_id"] = query_df["user_id"].astype(str)
        if "device_id" in query_df.columns:
            query_df["device_id"] = query_df["device_id"].astype(str)

        logger.info(
            f"Performing point-in-time historical join across {len(query_df):,} entity records..."
        )
        hist_response = self._store.get_historical_features(
            entity_df=query_df,
            features=feature_list,
        )
        return hist_response.to_df()

    def get_online_features(
        self,
        entity_rows: list[dict[str, Any]],
        features: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieves real-time features from the online store for inference."""
        feature_list = features or DEFAULT_FEAST_FEATURES
        response = self._store.get_online_features(
            features=feature_list,
            entity_rows=entity_rows,
        )
        return response.to_df().to_dict(orient="records")

    def get_online_features_single(
        self,
        user_id: str,
        device_id: str,
        features: list[str] | None = None,
    ) -> dict[str, Any]:
        """Low-latency convenience lookup for a single transaction scoring request."""
        results = self.get_online_features(
            entity_rows=[{"user_id": str(user_id), "device_id": str(device_id)}],
            features=features,
        )
        return results[0] if results else {}
