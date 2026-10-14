"""Decision engine categorizing risk scores into operational fraud decisions."""

from dataclasses import dataclass
from enum import StrEnum

from ml.config import settings


class Decision(StrEnum):
    """Operational decisions for fraud management."""

    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


@dataclass(frozen=True)
class DecisionResult:
    """Outcome of the fraud evaluation decision."""

    decision: Decision
    fraud_probability: float
    risk_score: float
    review_threshold: float
    block_threshold: float


class DecisionEngine:
    """Evaluates fraud probabilities against configurable operational thresholds."""

    def __init__(
        self,
        review_threshold: float | None = None,
        block_threshold: float | None = None,
    ) -> None:
        self.review_threshold = (
            review_threshold if review_threshold is not None else settings.decision.review_threshold
        )
        self.block_threshold = (
            block_threshold if block_threshold is not None else settings.decision.block_threshold
        )

        if self.review_threshold >= self.block_threshold:
            raise ValueError(
                f"review_threshold ({self.review_threshold}) must be strictly less than "
                f"block_threshold ({self.block_threshold})"
            )

    def decide(self, fraud_probability: float) -> DecisionResult:
        """Determines operational action based on calibrated probability."""
        prob = max(0.0, min(1.0, float(fraud_probability)))
        risk_score = round(prob * 100.0, 2)

        if prob >= self.block_threshold:
            decision = Decision.BLOCK
        elif prob >= self.review_threshold:
            decision = Decision.REVIEW
        else:
            decision = Decision.ALLOW

        return DecisionResult(
            decision=decision,
            fraud_probability=round(prob, 4),
            risk_score=risk_score,
            review_threshold=self.review_threshold,
            block_threshold=self.block_threshold,
        )


# Global decision engine instance
decision_engine = DecisionEngine()
