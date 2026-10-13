"""Training module for Logistic Regression baseline and LightGBM models with MLflow tracking."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.config import settings
from ml.evaluation.calibration import ModelCalibrator
from ml.evaluation.metrics import EvaluationReport, MetricsCalculator
from ml.logging import logger


@dataclass
class TrainedModelResult:
    """Encapsulates training outcomes, artifacts, and evaluation reports."""

    model_name: str
    model: Any
    calibrator: ModelCalibrator | None
    optimal_threshold: float
    train_report: EvaluationReport
    val_report: EvaluationReport
    test_report: EvaluationReport
    feature_importances: pd.DataFrame | None
    mlflow_run_id: str


class ModelTrainer:
    """Manages baseline and tree model training, probability calibration, and MLflow logging."""

    def __init__(self, metrics_calc: MetricsCalculator | None = None) -> None:
        self.metrics_calc = metrics_calc or MetricsCalculator(
            cost_fn=settings.model.cost_false_negative,
            cost_fp=settings.model.cost_false_positive,
        )

    def train_logistic_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> TrainedModelResult:
        """Trains a scaled Logistic Regression baseline."""
        logger.info("Training Logistic Regression baseline...")

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=settings.model.lr_max_iter,
                        C=settings.model.lr_c,
                        random_state=settings.data.random_seed,
                    ),
                ),
            ]
        )

        pipe.fit(X_train, y_train)

        # Calibrate on validation partition
        calibrator = ModelCalibrator(method="sigmoid")
        calibrator.fit_calibrator(pipe, X_val, y_val)

        # Predict calibrated probabilities
        val_probs = calibrator.predict_proba(X_val)
        optimal_th = self.metrics_calc.find_optimal_threshold(y_val.to_numpy(), val_probs)

        train_probs = calibrator.predict_proba(X_train)
        test_probs = calibrator.predict_proba(X_test)

        train_report = self.metrics_calc.evaluate(
            y_train.to_numpy(), train_probs, threshold=optimal_th
        )
        val_report = self.metrics_calc.evaluate(y_val.to_numpy(), val_probs, threshold=optimal_th)
        test_report = self.metrics_calc.evaluate(
            y_test.to_numpy(), test_probs, threshold=optimal_th
        )

        # Feature coefficients
        clf = pipe.named_steps["clf"]
        importances_df = (
            pd.DataFrame(
                {
                    "feature": X_train.columns,
                    "coefficient": clf.coef_[0],
                    "abs_importance": np.abs(clf.coef_[0]),
                }
            )
            .sort_values(by="abs_importance", ascending=False)
            .reset_index(drop=True)
        )

        run_id = self._log_to_mlflow(
            model_name="logistic_regression_baseline",
            model=pipe,
            params={
                "model_type": "LogisticRegression",
                "max_iter": settings.model.lr_max_iter,
                "C": settings.model.lr_c,
                "optimal_threshold": optimal_th,
            },
            train_report=train_report,
            val_report=val_report,
            test_report=test_report,
            feature_importances=importances_df,
        )

        return TrainedModelResult(
            model_name="logistic_regression",
            model=pipe,
            calibrator=calibrator,
            optimal_threshold=optimal_th,
            train_report=train_report,
            val_report=val_report,
            test_report=test_report,
            feature_importances=importances_df,
            mlflow_run_id=run_id,
        )

    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> TrainedModelResult:
        """Trains a gradient-boosted decision tree using LightGBM."""
        logger.info("Training LightGBM model...")

        # Calculate imbalance weight
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = float(neg_count / max(pos_count, 1))

        lgb_params = {
            "n_estimators": settings.model.lgb_n_estimators,
            "learning_rate": settings.model.lgb_learning_rate,
            "num_leaves": settings.model.lgb_num_leaves,
            "max_depth": settings.model.lgb_max_depth,
            "subsample": settings.model.lgb_subsample,
            "colsample_bytree": settings.model.lgb_colsample_bytree,
            "scale_pos_weight": scale_pos_weight,
            "random_state": settings.data.random_seed,
            "n_jobs": -1,
            "verbose": -1,
        }

        clf = lgb.LGBMClassifier(**lgb_params)
        clf.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

        # Calibrate using validation set
        calibrator = ModelCalibrator(method="isotonic")
        calibrator.fit_calibrator(clf, X_val, y_val)

        val_probs = calibrator.predict_proba(X_val)
        optimal_th = self.metrics_calc.find_optimal_threshold(y_val.to_numpy(), val_probs)

        train_probs = calibrator.predict_proba(X_train)
        test_probs = calibrator.predict_proba(X_test)

        train_report = self.metrics_calc.evaluate(
            y_train.to_numpy(), train_probs, threshold=optimal_th
        )
        val_report = self.metrics_calc.evaluate(y_val.to_numpy(), val_probs, threshold=optimal_th)
        test_report = self.metrics_calc.evaluate(
            y_test.to_numpy(), test_probs, threshold=optimal_th
        )

        # Feature importances
        importances_df = (
            pd.DataFrame(
                {
                    "feature": X_train.columns,
                    "importance": clf.feature_importances_,
                }
            )
            .sort_values(by="importance", ascending=False)
            .reset_index(drop=True)
        )

        run_id = self._log_to_mlflow(
            model_name="lightgbm_fraud_model",
            model=clf,
            params={
                **lgb_params,
                "optimal_threshold": optimal_th,
                "best_iteration": int(clf.best_iteration_ or 0),
            },
            train_report=train_report,
            val_report=val_report,
            test_report=test_report,
            feature_importances=importances_df,
        )

        return TrainedModelResult(
            model_name="lightgbm",
            model=clf,
            calibrator=calibrator,
            optimal_threshold=optimal_th,
            train_report=train_report,
            val_report=val_report,
            test_report=test_report,
            feature_importances=importances_df,
            mlflow_run_id=run_id,
        )

    def _log_to_mlflow(
        self,
        model_name: str,
        model: Any,
        params: dict[str, Any],
        train_report: EvaluationReport,
        val_report: EvaluationReport,
        test_report: EvaluationReport,
        feature_importances: pd.DataFrame | None,
    ) -> str:
        """Logs parameters, metrics across splits, feature importances, and model artifact to MLflow."""
        Path("artifacts").mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
        mlflow.set_experiment(settings.mlflow.experiment_name)

        with mlflow.start_run(run_name=model_name) as run:
            # Parameters
            mlflow.log_params(params)
            mlflow.set_tag("pipeline", "fraudshield-phase1")

            # Metrics for each split
            for k, v in train_report.to_dict().items():
                mlflow.log_metric(f"train_{k}", float(v))
            for k, v in val_report.to_dict().items():
                mlflow.log_metric(f"val_{k}", float(v))
            for k, v in test_report.to_dict().items():
                mlflow.log_metric(f"test_{k}", float(v))

            # Feature importances artifact
            if feature_importances is not None:
                importance_path = f"artifacts/{model_name}_feature_importance.csv"
                pd.DataFrame(feature_importances).to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path)

            # Model artifact
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                serialization_format="cloudpickle",
            )

            logger.info(
                f"Logged {model_name} to MLflow run [{run.info.run_id}] | "
                f"Test PR-AUC: {test_report.pr_auc:.4f} | Test ROC-AUC: {test_report.roc_auc:.4f} | "
                f"Expected Cost: ${test_report.expected_cost:,.2f}"
            )
            return run.info.run_id
