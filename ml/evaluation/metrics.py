"""Comprehensive evaluation metrics for imbalanced fraud classification."""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class EvaluationReport:
    """Comprehensive evaluation metrics container."""

    pr_auc: float
    roc_auc: float
    brier_score: float
    threshold: float
    precision: float
    recall: float
    f1: float
    fpr: float
    fnr: float
    tp: int
    fp: int
    tn: int
    fn: int
    expected_cost: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "pr_auc": round(self.pr_auc, 5),
            "roc_auc": round(self.roc_auc, 5),
            "brier_score": round(self.brier_score, 5),
            "threshold": round(self.threshold, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "fpr": round(self.fpr, 5),
            "fnr": round(self.fnr, 5),
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "expected_cost": round(self.expected_cost, 2),
        }


class MetricsCalculator:
    """Computes fraud metrics, confusion matrices, and cost-weighted trade-offs."""

    def __init__(self, cost_fn: float = 500.0, cost_fp: float = 20.0) -> None:
        self.cost_fn = cost_fn
        self.cost_fp = cost_fp

    def evaluate(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.5,
    ) -> EvaluationReport:
        """Evaluates model performance given ground truth and predicted fraud probabilities."""
        pr_auc = float(average_precision_score(y_true, y_prob))
        roc_auc = float(roc_auc_score(y_true, y_prob))
        brier = float(brier_score_loss(y_true, y_prob))

        y_pred = (y_prob >= threshold).astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

        expected_cost = float((fn * self.cost_fn) + (fp * self.cost_fp))

        return EvaluationReport(
            pr_auc=pr_auc,
            roc_auc=roc_auc,
            brier_score=brier,
            threshold=threshold,
            precision=precision,
            recall=recall,
            f1=f1,
            fpr=fpr,
            fnr=fnr,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            expected_cost=expected_cost,
        )

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        grid_steps: int = 100,
    ) -> float:
        """Finds the decision threshold minimizing total business cost."""
        thresholds = np.linspace(0.01, 0.99, grid_steps)
        best_threshold = 0.5
        min_cost = float("inf")

        for th in thresholds:
            y_pred = (y_prob >= th).astype(int)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            fp, fn = int(cm[0, 1]), int(cm[1, 0])
            cost = (fn * self.cost_fn) + (fp * self.cost_fp)
            if cost < min_cost:
                min_cost = cost
                best_threshold = float(th)

        return best_threshold
