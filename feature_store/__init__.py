"""Feast Declarative Feature Store package for FraudShield."""

from feature_store.client import FeatureStoreClient
from feature_store.dataset_builder import (
    build_and_save_feature_tables,
    build_device_features,
    build_user_features,
)
from feature_store.features import (
    all_entities,
    all_feature_views,
    device_entity,
    device_reputation_fv,
    user_behavior_fv,
    user_entity,
)
from feature_store.materialize import run_materialization

__all__ = [
    "FeatureStoreClient",
    "all_entities",
    "all_feature_views",
    "build_and_save_feature_tables",
    "build_device_features",
    "build_user_features",
    "device_entity",
    "device_reputation_fv",
    "run_materialization",
    "user_behavior_fv",
    "user_entity",
]
