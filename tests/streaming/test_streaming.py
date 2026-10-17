"""Integration tests for streaming components: storage, processor, and DLQ routing."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from apps.stream_processor.processor import FraudStreamProcessor
from apps.stream_processor.producer import json_serializer
from apps.stream_processor.storage import TransactionStorage
from ml.inference.decision_engine import Decision
from ml.inference.model_loader import model_loader
from ml.inference.predictor import ScoredTransaction, TransactionPredictor


@pytest.fixture(scope="module", autouse=True)
def ensure_model_loaded():
    """Ensures model is loaded in memory for all streaming tests."""
    if not model_loader.is_loaded:
        model_loader.load()


@pytest.fixture
def temp_storage(tmp_path: Path):
    """Creates a temporary isolated SQLite database for testing."""
    db_file = tmp_path / "test_scored.db"
    store = TransactionStorage(postgres_url=None, sqlite_path=db_file)
    store._using_sqlite = True
    store._init_sqlite_schema()
    return store


@pytest.mark.anyio
async def test_storage_save_and_retrieve(temp_storage: TransactionStorage) -> None:
    scored = ScoredTransaction(
        transaction_id="tx_stream_test_001",
        fraud_probability=0.885,
        risk_score=88.5,
        decision=Decision.BLOCK,
        model_version="fraud-model-v1",
        latency_ms=12.4,
        top_risk_factors=["new_device", "high_amount_deviation"],
    )

    await temp_storage.save_transaction(
        scored=scored,
        user_id="user_stream_01",
        merchant_id="merchant_01",
        amount=54000.0,
        currency="INR",
        timestamp=datetime.now(UTC),
    )

    # Verify retrieval
    retrieved = await temp_storage.get_transaction("tx_stream_test_001")
    assert retrieved is not None
    assert retrieved["transaction_id"] == "tx_stream_test_001"
    assert retrieved["user_id"] == "user_stream_01"
    assert retrieved["decision"] == "BLOCK"
    assert retrieved["risk_score"] == 88.5
    assert retrieved["top_risk_factors"] == ["new_device", "high_amount_deviation"]

    # Verify count
    count = await temp_storage.count_transactions()
    assert count == 1


@pytest.mark.anyio
async def test_processor_valid_record(temp_storage: TransactionStorage) -> None:
    pred = TransactionPredictor()
    processor = FraudStreamProcessor(
        pred=pred,
        db_storage=temp_storage,
        loader=model_loader,
    )

    valid_record = {
        "transaction_id": "tx_proc_001",
        "user_id": "user_100",
        "merchant_id": "merchant_200",
        "device_id": "device_300",
        "amount": 450.0,
        "currency": "INR",
        "timestamp": datetime.now(UTC).isoformat(),
        "ip_address": "192.168.1.50",
        "merchant_category": "retail",
    }

    result = await processor.process_record(valid_record)
    assert result["transaction_id"] == "tx_proc_001"
    assert "fraud_probability" in result
    assert result["decision"] in ("ALLOW", "REVIEW", "BLOCK")
    assert processor.processed_count == 1
    assert processor.dlq_count == 0

    # Verify saved in storage
    saved = await temp_storage.get_transaction("tx_proc_001")
    assert saved is not None
    assert saved["amount"] == 450.0


@pytest.mark.anyio
async def test_processor_invalid_record_routes_to_dlq(temp_storage: TransactionStorage) -> None:
    pred = TransactionPredictor()
    processor = FraudStreamProcessor(
        pred=pred,
        db_storage=temp_storage,
        loader=model_loader,
    )

    # Invalid: negative amount violates TransactionRecord schema
    invalid_record = {
        "transaction_id": "tx_invalid_dlq_002",
        "user_id": "user_100",
        "merchant_id": "merchant_200",
        "amount": -999.0,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    result = await processor.process_record(invalid_record)
    assert result["status"] == "DLQ"
    assert "reason" in result
    assert processor.dlq_count == 1
    assert processor.processed_count == 0

    # Verify not stored in regular storage
    saved = await temp_storage.get_transaction("tx_invalid_dlq_002")
    assert saved is None


def test_producer_json_serializer() -> None:
    now = datetime(2026, 10, 14, 12, 0, 0, tzinfo=UTC)
    payload = {"timestamp": now, "amount": 100.5}

    serialized = json_serializer(payload)
    assert isinstance(serialized, bytes)
    deserialized = json.loads(serialized.decode("utf-8"))
    assert deserialized["amount"] == 100.5
    assert "2026-10-14" in deserialized["timestamp"]
