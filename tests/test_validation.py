"""Unit tests for data validation module."""

from datetime import datetime

import pandas as pd

from ml.data.validation import DataValidator


def test_validator_with_valid_data() -> None:
    validator = DataValidator()
    df = pd.DataFrame(
        {
            "transaction_id": ["tx_1", "tx_2"],
            "user_id": ["u_1", "u_2"],
            "merchant_id": ["m_1", "m_2"],
            "device_id": ["d_1", "d_2"],
            "amount": [100.5, 250.0],
            "timestamp": [datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 1, 11, 0)],
            "merchant_category": ["retail", "grocery"],
            "is_fraud": [0, 1],
        }
    )
    is_valid, report = validator.validate(df)
    assert is_valid is True
    assert report.total_records == 2
    assert report.fraud_count == 1
    assert report.negative_amounts == 0
    assert len(report.errors) == 0


def test_validator_detects_negative_amounts() -> None:
    validator = DataValidator()
    df = pd.DataFrame(
        {
            "transaction_id": ["tx_1", "tx_2"],
            "user_id": ["u_1", "u_2"],
            "merchant_id": ["m_1", "m_2"],
            "device_id": ["d_1", "d_2"],
            "amount": [-50.0, 100.0],
            "timestamp": [datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 1, 11, 0)],
            "merchant_category": ["retail", "grocery"],
            "is_fraud": [0, 0],
        }
    )
    is_valid, report = validator.validate(df)
    assert is_valid is False
    assert report.negative_amounts == 1
    assert any("amount <= 0" in err for err in report.errors)


def test_validator_detects_duplicates() -> None:
    validator = DataValidator()
    df = pd.DataFrame(
        {
            "transaction_id": ["tx_1", "tx_1"],
            "user_id": ["u_1", "u_2"],
            "merchant_id": ["m_1", "m_2"],
            "device_id": ["d_1", "d_2"],
            "amount": [100.0, 200.0],
            "timestamp": [datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 1, 11, 0)],
            "merchant_category": ["retail", "grocery"],
            "is_fraud": [0, 0],
        }
    )
    is_valid, report = validator.validate(df)
    assert is_valid is False
    assert report.duplicate_ids == 1
