"""Integration tests for the FastAPI fraud scoring endpoints."""

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app
from ml.inference.decision_engine import Decision
from ml.inference.model_loader import model_loader


@pytest.fixture(scope="module", autouse=True)
def ensure_model_loaded():
    """Ensures model is loaded in memory for all API tests."""
    if not model_loader.is_loaded:
        model_loader.load()


@pytest.fixture
def client():
    """Test client for FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "version" in data


def test_readiness_check(client: TestClient) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "model_name" in data
    assert data["features_count"] == 12


def test_score_legitimate_transaction(client: TestClient) -> None:
    payload = {
        "transaction_id": "tx_test_legit_001",
        "user_id": "user_0001",
        "merchant_id": "merchant_0001",
        "device_id": "device_00001",
        "amount": 250.0,
        "currency": "INR",
        "timestamp": "2026-10-13T14:30:00Z",  # Daytime
        "ip_address": "192.168.1.100",
        "merchant_category": "grocery",
    }
    response = client.post("/v1/transactions/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "tx_test_legit_001"
    assert "fraud_probability" in data
    assert "risk_score" in data
    assert data["decision"] in (Decision.ALLOW, Decision.REVIEW, Decision.BLOCK)
    assert data["latency_ms"] >= 0.0
    assert isinstance(data["top_risk_factors"], list)


def test_score_suspicious_spike_transaction(client: TestClient) -> None:
    payload = {
        "transaction_id": "tx_test_fraud_002",
        "user_id": "user_9999",
        "merchant_id": "merchant_9999",
        "device_id": "device_brand_new_suspicious",
        "amount": 95000.0,  # High monetary spike
        "currency": "INR",
        "timestamp": "2026-10-13T03:15:00Z",  # 3 AM night
        "ip_address": "192.168.1.250",
        "merchant_category": "crypto_exchange",
    }
    response = client.post("/v1/transactions/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "tx_test_fraud_002"
    assert data["fraud_probability"] > 0.0
    # Check that explainability risk factors are surfaced
    assert any(
        factor in data["top_risk_factors"]
        for factor in ["new_device", "unusual_transaction_time", "large_monetary_amount"]
    )


def test_get_scored_transaction_by_id(client: TestClient) -> None:
    tx_id = "tx_test_lookup_003"
    payload = {
        "transaction_id": tx_id,
        "user_id": "user_0005",
        "merchant_id": "merchant_0010",
        "device_id": "device_00005",
        "amount": 1200.0,
        "currency": "INR",
        "timestamp": "2026-10-13T12:00:00Z",
        "ip_address": "192.168.1.15",
        "merchant_category": "retail",
    }
    # Score it
    score_res = client.post("/v1/transactions/score", json=payload)
    assert score_res.status_code == 200

    # Retrieve it
    get_res = client.get(f"/v1/transactions/{tx_id}")
    assert get_res.status_code == 200
    retrieved = get_res.json()
    assert retrieved["transaction_id"] == tx_id
    assert retrieved["risk_score"] == score_res.json()["risk_score"]


def test_get_unknown_transaction_returns_404(client: TestClient) -> None:
    response = client.get("/v1/transactions/non_existent_tx_999999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_negative_amount_fails_validation(client: TestClient) -> None:
    payload = {
        "transaction_id": "tx_invalid_001",
        "user_id": "user_0001",
        "merchant_id": "merchant_0001",
        "device_id": "device_00001",
        "amount": -50.0,  # Negative amount rejected
        "timestamp": "2026-10-13T14:30:00Z",
        "ip_address": "192.168.1.100",
        "merchant_category": "grocery",
    }
    response = client.post("/v1/transactions/score", json=payload)
    assert response.status_code == 422


def test_get_current_model_metadata(client: TestClient) -> None:
    response = client.get("/v1/models/current")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "optimal_threshold" in data
    assert len(data["feature_names"]) == 12
    assert "metrics" in data


def test_get_operational_metrics(client: TestClient) -> None:
    response = client.get("/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_scored" in data
    assert data["total_scored"] >= 1
    assert "average_latency_ms" in data
    assert "decisions" in data
    assert "rates" in data
