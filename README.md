# FraudShield

> **Real-Time Fraud Detection & Continuous ML Platform**

FraudShield is a production-grade, end-to-end real-time fraud detection platform that analyzes financial transactions using behavioral, tabular, and graph-based machine learning, serves low-latency risk scores, monitors drift, and continuously retrains models.

---

## Architecture Overview

```text
Incoming Transaction 
        │
        ▼
┌───────────────────────────┐
│     FastAPI Gateway       │  <-- POST /v1/transactions/score (<30ms)
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│   Online Feature Engine   │  <-- 12 Behavioral & Velocity Features
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│   Calibrated Model Engine │  <-- LightGBM + Isotonic Probability Calibration
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│      Decision Engine      │  <-- Configurable Thresholds: ALLOW / REVIEW / BLOCK
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│ Explainability & Metrics  │  <-- Top Risk Factors + Real-Time Telemetry
└───────────────────────────┘
```

---

## Current Status: Implemented Phases

### Phase 0: Project Foundation
* **Modern Stack:** Python 3.12+, `uv` package management, strict type hints, and PEP 621 `pyproject.toml`.
* **Code Hygiene:** `ruff` (linter & formatter) targeting Python 3.12, `pytest` test harness.
* **Central Configuration:** Type-safe settings via `pydantic-settings` with immutable `BaseModel` schemas.
* **Logging:** Structured console and file logging via `rich.logging.RichHandler`.

### Phase 1: Offline ML Pipeline & Calibration
* **Data Ingestion & Generation:** Synthetic streaming transaction generator simulating 4 realistic fraud attack modes (amount spikes, velocity bursts, device hijacking, night anomalies).
* **Data Integrity Validation:** `DataValidator` enforcing schema conformity, zero negative amounts, and zero duplicate IDs.
* **Zero-Leakage Temporal Splitting:** Strict chronological partitions (Train: 70%, Val: 15%, Test: 15%) with runtime assertion gates preventing lookahead bias.
* **Leak-Free Feature Engineering:** 12 backward-looking behavioral, velocity, cyclical, and identity features.
* **Model Training:** Scaled `LogisticRegression` baseline and cost-weighted `LightGBM` (`scale_pos_weight`, early stopping).
* **Probability Calibration:** Native Platt scaling (Sigmoid) and Isotonic regression fitted strictly on validation data.
* **Cost-Weighted Threshold Optimization:** Grid optimization minimizing business losses ($Cost(FN) = \$500$ vs $Cost(FP) = \$20$).
* **Experiment Tracking:** MLflow tracking logging parameters, metrics across all splits, feature importances, and serialized model artifacts.

### Phase 2: Low-Latency Inference API & Docker
* **FastAPI Application:** Modern `@asynccontextmanager` `lifespan` handler preloading model artifacts into memory on startup.
* **Thread-Safe Model Loader:** Caches calibrated LightGBM booster and metadata in RAM.
* **Decision Engine:** Evaluates fraud probabilities against configurable operational thresholds (`ALLOW`, `REVIEW`, `BLOCK`).
* **Explainability Module:** Surfaces contextual `top_risk_factors` (e.g., `new_device`, `high_amount_deviation`, `unusual_transaction_time`).
* **Containerization:** Multi-stage `Dockerfile` (using `uv` for sub-second image builds) and local `docker-compose.yml`.

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **`GET`** | `/health` | Liveness probe verifying process responsiveness |
| **`GET`** | `/ready` | Readiness probe confirming model artifact is loaded in RAM |
| **`POST`** | `/v1/transactions/score` | Scores an incoming transaction, evaluates risk, and returns decision |
| **`GET`** | `/v1/transactions/{id}` | Audit lookup for a previously scored transaction |
| **`GET`** | `/v1/models/current` | Returns active model metadata, feature names, and test benchmarks |
| **`GET`** | `/v1/metrics` | Returns operational telemetry (total scored, latency, decision rates) |

### Sample Scoring Request

```bash
curl -X POST http://localhost:8000/v1/transactions/score \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "tx_100001",
    "user_id": "user_482",
    "merchant_id": "merchant_183",
    "amount": 87500.0,
    "currency": "INR",
    "timestamp": "2026-10-13T03:14:22Z",
    "device_id": "device_new_99",
    "ip_address": "192.168.1.45",
    "merchant_category": "crypto_exchange",
    "location": {
      "country": "IN",
      "city": "Mumbai"
    }
  }'
```

### Sample Scoring Response

```json
{
  "transaction_id": "tx_100001",
  "fraud_probability": 0.9412,
  "risk_score": 94.12,
  "decision": "BLOCK",
  "model_version": "fraud-model-v1",
  "latency_ms": 14.82,
  "top_risk_factors": [
    "new_device",
    "high_amount_deviation",
    "unusual_transaction_time",
    "large_monetary_amount",
    "high_risk_merchant_category"
  ]
}
```

---

## Quickstart Guide

### 1. Environment Setup
```bash
# Create virtual environment with uv
uv venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Install dependencies in editable mode with dev tools
uv pip install -e ".[dev]"
```

### 2. Run Offline Training Pipeline (Phase 1)
Generates synthetic transactions, validates data, fits leak-free feature pipelines, trains LightGBM, calibrates probabilities, logs to MLflow, and exports `artifacts/models/best_fraud_model.joblib`:
```bash
python -m ml.pipelines.training_pipeline
```

### 3. Run the Inference API (Phase 2)
```bash
# Start locally with Uvicorn
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API documentation is accessible at `http://localhost:8000/docs`.

### 4. Run via Docker Compose
```bash
docker compose up --build
```

### 5. Run Test Suite
```bash
# Run all unit and API integration tests
pytest

# Run linter and formatter checks
ruff check .
ruff format --check .
```
