"""Central configuration for FraudShield using Pydantic Settings."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataConfig(BaseModel):
    """Configuration for data paths and simulation."""

    model_config = ConfigDict(frozen=True)

    data_dir: Path = Field(default=Path("data"))
    raw_data_path: Path = Field(default=Path("data/raw/transactions.parquet"))
    sample_size: int = Field(default=50_000, description="Synthetic samples if generating data")
    fraud_rate: float = Field(default=0.03, description="Base fraud occurrence probability")
    random_seed: int = Field(default=42)


class SplitConfig(BaseModel):
    """Temporal split boundary configurations."""

    model_config = ConfigDict(frozen=True)

    train_ratio: float = Field(default=0.70)
    val_ratio: float = Field(default=0.15)
    test_ratio: float = Field(default=0.15)


class ModelParams(BaseModel):
    """Model hyperparameters and cost matrix weights."""

    model_config = ConfigDict(frozen=True)

    # Cost Matrix Weights
    cost_false_negative: float = Field(default=500.0, description="Cost of missed fraud")
    cost_false_positive: float = Field(default=20.0, description="Cost of false alarm/friction")

    # Logistic Regression
    lr_max_iter: int = Field(default=1000)
    lr_c: float = Field(default=1.0)

    # LightGBM
    lgb_n_estimators: int = Field(default=300)
    lgb_learning_rate: float = Field(default=0.05)
    lgb_num_leaves: int = Field(default=31)
    lgb_max_depth: int = Field(default=6)
    lgb_subsample: float = Field(default=0.8)
    lgb_colsample_bytree: float = Field(default=0.8)


class MLflowConfig(BaseModel):
    """MLflow tracking setup."""

    model_config = ConfigDict(frozen=True)

    tracking_uri: str = Field(default="sqlite:///mlflow.db")
    experiment_name: str = Field(default="fraudshield-offline-ml")
    artifact_location: Path = Field(default=Path("artifacts/mlflow"))


class DecisionConfig(BaseModel):
    """Configurable thresholds for fraud decision engine."""

    model_config = ConfigDict(frozen=True)

    # Decisions: ALLOW if prob < review_threshold, REVIEW if review_threshold <= prob < block_threshold, BLOCK if prob >= block_threshold
    review_threshold: float = Field(
        default=0.03, description="Risk probability above which transactions are sent to REVIEW"
    )
    block_threshold: float = Field(
        default=0.20,
        description="Risk probability above which transactions are BLOCKED immediately",
    )


class APIConfig(BaseModel):
    """FastAPI service configuration."""

    model_config = ConfigDict(frozen=True)

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    model_path: Path = Field(default=Path("artifacts/models/best_fraud_model.joblib"))
    model_version: str = Field(default="fraud-model-v1")


class KafkaConfig(BaseModel):
    """Kafka / Redpanda stream configuration."""

    model_config = ConfigDict(frozen=True)

    bootstrap_servers: str = Field(default="localhost:9092")
    topic_raw: str = Field(default="transactions.raw")
    topic_scored: str = Field(default="transactions.scored")
    topic_dlq: str = Field(default="transactions.dlq")
    consumer_group: str = Field(default="fraudshield-scorer-group")
    auto_offset_reset: str = Field(default="earliest")


class DatabaseConfig(BaseModel):
    """PostgreSQL and local storage configuration."""

    model_config = ConfigDict(frozen=True)

    postgres_url: str = Field(
        default="postgresql://fraudshield:fraudshield@localhost:5432/fraudshield_db"
    )
    sqlite_fallback_path: Path = Field(default=Path("artifacts/scored_transactions.db"))


class Settings(BaseSettings):
    """Global FraudShield system configuration."""

    model_config = SettingsConfigDict(
        env_prefix="FRAUDSHIELD_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")

    data: DataConfig = Field(default_factory=DataConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    model: ModelParams = Field(default_factory=ModelParams)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)


# Global settings singleton
settings = Settings()
