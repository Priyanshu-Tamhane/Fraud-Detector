"""Pydantic v2 schemas for the FraudShield API."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ml.inference.decision_engine import Decision


class LocationSchema(BaseModel):
    """Geographic transaction location."""

    model_config = ConfigDict(extra="ignore")

    country: str = Field(default="IN", description="ISO country code")
    city: str = Field(default="Mumbai", description="City name")


class TransactionScoreRequest(BaseModel):
    """Incoming transaction scoring payload."""

    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "example": {
                "transaction_id": "tx_100001",
                "user_id": "user_482",
                "merchant_id": "merchant_183",
                "amount": 87500.0,
                "currency": "INR",
                "timestamp": "2026-10-05T03:14:22Z",
                "device_id": "device_72",
                "ip_address": "10.20.30.40",
                "merchant_category": "electronics",
                "location": {
                    "country": "IN",
                    "city": "Mumbai",
                },
            }
        },
    )

    transaction_id: str = Field(..., description="Unique transaction reference ID")
    user_id: str = Field(..., description="Account or user identifier")
    merchant_id: str = Field(..., description="Merchant identifier")
    amount: float = Field(..., gt=0.0, description="Positive transaction monetary amount")
    currency: str = Field(default="INR", description="Currency code (e.g. INR, USD)")
    timestamp: datetime = Field(..., description="UTC ISO-8601 transaction timestamp")
    device_id: str = Field(..., description="Client device identifier")
    ip_address: str = Field(..., description="Client IP address")
    merchant_category: str = Field(..., description="Merchant category code or description")
    location: LocationSchema | None = Field(default=None, description="Optional location object")


class TransactionScoreResponse(BaseModel):
    """Scored transaction result and decision."""

    model_config = ConfigDict(frozen=True)

    transaction_id: str
    fraud_probability: float
    risk_score: float
    decision: Decision
    model_version: str
    latency_ms: float
    top_risk_factors: list[str]


class HealthResponse(BaseModel):
    """API health status."""

    status: str
    model_loaded: bool
    version: str
    timestamp: datetime


class ModelCurrentResponse(BaseModel):
    """Currently deployed model metadata and benchmark metrics."""

    model_name: str
    model_version: str
    optimal_threshold: float
    review_threshold: float
    block_threshold: float
    feature_names: list[str]
    metrics: dict[str, Any]
