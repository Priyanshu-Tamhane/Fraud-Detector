"""Declarative Feast Feature Store definitions for FraudShield."""

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64

from ml.config import settings

# 1. Declarative Entities
user_entity = Entity(
    name="user_id",
    value_type=ValueType.STRING,
    join_keys=["user_id"],
    description="Unique user account identifier",
)

device_entity = Entity(
    name="device_id",
    value_type=ValueType.STRING,
    join_keys=["device_id"],
    description="Unique hardware device identifier",
)

# 2. Batch Offline Data Sources
# Paths are resolved to absolute to guarantee cross-directory resolution
user_source_path = str(Path(settings.feast.user_features_parquet).resolve())
device_source_path = str(Path(settings.feast.device_features_parquet).resolve())

user_batch_source = FileSource(
    path=user_source_path,
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

device_batch_source = FileSource(
    path=device_source_path,
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# 3. Declarative Feature Views
user_behavior_fv = FeatureView(
    name="user_behavior_fv",
    entities=[user_entity],
    ttl=timedelta(days=90),
    schema=[
        Field(
            name="user_prior_count",
            dtype=Int64,
            description="Historical transaction count for user prior to event",
        ),
        Field(
            name="user_prior_mean_amount",
            dtype=Float32,
            description="Historical average spend for user prior to event",
        ),
        Field(
            name="user_prior_sum",
            dtype=Float32,
            description="Historical total spend for user prior to event",
        ),
        Field(
            name="user_fraud_count",
            dtype=Int64,
            description="Past confirmed fraud count for user prior to event",
        ),
    ],
    online=True,
    source=user_batch_source,
    description="User behavioral and spending velocity profile",
)

device_reputation_fv = FeatureView(
    name="device_reputation_fv",
    entities=[device_entity],
    ttl=timedelta(days=90),
    schema=[
        Field(
            name="device_unique_users_seen",
            dtype=Int64,
            description="Distinct accounts using this device prior to event",
        ),
        Field(
            name="is_device_flagged",
            dtype=Int64,
            description="Whether device was flagged for fraud prior to event",
        ),
    ],
    online=True,
    source=device_batch_source,
    description="Device hardware reputation and account multiplexing profile",
)

all_entities = [user_entity, device_entity]
all_feature_views = [user_behavior_fv, device_reputation_fv]
