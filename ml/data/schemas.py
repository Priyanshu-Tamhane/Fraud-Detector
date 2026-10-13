"""Data schemas for transactions and validation."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class TransactionRecord(BaseModel):
    """Pydantic v2 schema representing a financial transaction."""

    model_config = ConfigDict(extra="ignore")

    transaction_id: str = Field(..., description="Unique transaction ID")
    user_id: str = Field(..., description="Unique user/account ID")
    merchant_id: str = Field(..., description="Unique merchant ID")
    device_id: str = Field(..., description="Unique device ID")
    amount: float = Field(..., gt=0.0, description="Transaction monetary amount")
    currency: str = Field(default="INR", description="Three-letter currency code")
    timestamp: datetime = Field(..., description="UTC transaction timestamp")
    ip_address: str = Field(..., description="Client IP address")
    merchant_category: str = Field(..., description="Category: e.g. retail, grocery, electronics")
    country: str = Field(default="IN", description="Two-letter country code")
    city: str = Field(default="Mumbai", description="City of transaction")
    is_fraud: int = Field(default=0, ge=0, le=1, description="Binary ground truth label")
