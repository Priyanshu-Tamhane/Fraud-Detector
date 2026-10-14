"""FraudShield FastAPI service for low-latency transaction scoring and observability."""

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from apps.api.schemas import (
    HealthResponse,
    ModelCurrentResponse,
    TransactionScoreRequest,
    TransactionScoreResponse,
)
from ml.config import settings
from ml.inference.decision_engine import Decision, decision_engine
from ml.inference.model_loader import model_loader
from ml.inference.predictor import ScoredTransaction, predictor
from ml.logging import logger

# In-memory store for scored transactions and basic API metrics
_scored_history: dict[str, ScoredTransaction] = {}
_metrics = {
    "total_requests": 0,
    "allow_count": 0,
    "review_count": 0,
    "block_count": 0,
    "total_latency_ms": 0.0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: loads model into memory before accepting traffic."""
    logger.info("Initializing FraudShield Inference Engine...")
    try:
        model_loader.load()
        logger.info("Model loaded successfully. Service is READY.")
    except Exception as e:
        logger.warning(
            f"Could not preload model artifact on startup: {e}. "
            "Ensure training pipeline has completed before scoring transactions."
        )
    yield
    logger.info("Shutting down FraudShield Inference Engine.")


app = FastAPI(
    title="FraudShield Inference API",
    description="Low-latency real-time financial fraud risk scoring and decision platform.",
    version="0.2.0",
    lifespan=lifespan,
)

# Standard CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Observability"])
async def health_check() -> HealthResponse:
    """Liveness probe indicating the service process is responsive."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_loader.is_loaded,
        version=settings.api.model_version,
        timestamp=datetime.now(UTC),
    )


@app.get("/ready", tags=["Observability"])
async def readiness_check() -> JSONResponse:
    """Readiness probe indicating if the service is primed to receive scoring requests."""
    if not model_loader.is_loaded:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_ready",
                "reason": "Model artifact is not loaded in memory.",
            },
        )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "ready",
            "model_name": model_loader.model_name,
            "features_count": len(model_loader.feature_names),
        },
    )


@app.post(
    "/v1/transactions/score",
    response_model=TransactionScoreResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
async def score_transaction(payload: TransactionScoreRequest) -> TransactionScoreResponse:
    """Scores an incoming transaction in real-time, calculates risk, and returns ALLOW/REVIEW/BLOCK."""
    if not model_loader.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready. Ensure model artifacts are present.",
        )

    try:
        scored: ScoredTransaction = predictor.predict(
            transaction_id=payload.transaction_id,
            user_id=payload.user_id,
            merchant_id=payload.merchant_id,
            device_id=payload.device_id,
            amount=payload.amount,
            timestamp=payload.timestamp,
            merchant_category=payload.merchant_category,
        )

        # Store in historical cache
        _scored_history[payload.transaction_id] = scored

        # Update metrics
        _metrics["total_requests"] += 1
        _metrics["total_latency_ms"] += scored.latency_ms
        if scored.decision == Decision.ALLOW:
            _metrics["allow_count"] += 1
        elif scored.decision == Decision.REVIEW:
            _metrics["review_count"] += 1
        elif scored.decision == Decision.BLOCK:
            _metrics["block_count"] += 1

        return TransactionScoreResponse(
            transaction_id=scored.transaction_id,
            fraud_probability=scored.fraud_probability,
            risk_score=scored.risk_score,
            decision=scored.decision,
            model_version=scored.model_version,
            latency_ms=scored.latency_ms,
            top_risk_factors=scored.top_risk_factors,
        )
    except Exception as exc:
        logger.error(f"Inference error scoring transaction '{payload.transaction_id}': {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failure: {exc}",
        ) from exc


@app.get(
    "/v1/transactions/{transaction_id}",
    response_model=TransactionScoreResponse,
    tags=["Inference"],
)
async def get_scored_transaction(transaction_id: str) -> TransactionScoreResponse:
    """Retrieves decision and risk score for a previously evaluated transaction."""
    scored = _scored_history.get(transaction_id)
    if not scored:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction '{transaction_id}' not found in audit history.",
        )
    return TransactionScoreResponse(
        transaction_id=scored.transaction_id,
        fraud_probability=scored.fraud_probability,
        risk_score=scored.risk_score,
        decision=scored.decision,
        model_version=scored.model_version,
        latency_ms=scored.latency_ms,
        top_risk_factors=scored.top_risk_factors,
    )


@app.get("/v1/models/current", response_model=ModelCurrentResponse, tags=["Model Registry"])
async def get_current_model() -> ModelCurrentResponse:
    """Returns active model metadata, thresholds, and performance metrics."""
    if not model_loader.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )
    return ModelCurrentResponse(
        model_name=model_loader.model_name,
        model_version=settings.api.model_version,
        optimal_threshold=model_loader.optimal_threshold,
        review_threshold=decision_engine.review_threshold,
        block_threshold=decision_engine.block_threshold,
        feature_names=model_loader.feature_names,
        metrics=model_loader.metrics,
    )


@app.get("/v1/metrics", tags=["Observability"])
async def get_operational_metrics() -> dict[str, Any]:
    """Returns runtime scoring latency and decision distributions."""
    total = _metrics["total_requests"]
    avg_latency = (_metrics["total_latency_ms"] / total) if total > 0 else 0.0

    return {
        "total_scored": total,
        "average_latency_ms": round(avg_latency, 2),
        "decisions": {
            "ALLOW": _metrics["allow_count"],
            "REVIEW": _metrics["review_count"],
            "BLOCK": _metrics["block_count"],
        },
        "rates": {
            "allow_rate": round(_metrics["allow_count"] / total, 4) if total > 0 else 0.0,
            "review_rate": round(_metrics["review_count"] / total, 4) if total > 0 else 0.0,
            "block_rate": round(_metrics["block_count"] / total, 4) if total > 0 else 0.0,
        },
    }
