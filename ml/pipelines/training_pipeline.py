"""Reproducible end-to-end offline ML training and evaluation pipeline."""

from pathlib import Path

import joblib
import pandas as pd
from rich.console import Console
from rich.table import Table

from ml.config import settings
from ml.data.generator import TransactionGenerator
from ml.data.validation import DataValidator
from ml.features.pipeline import FeaturePipeline
from ml.features.temporal_split import TemporalSplitter
from ml.logging import logger
from ml.training.trainer import ModelTrainer

console = Console()


def run_training_pipeline() -> None:
    """Executes the full Phase 1 offline ML pipeline."""
    logger.info("=" * 60)
    logger.info("Starting FraudShield Offline ML Training Pipeline")
    logger.info("=" * 60)

    # 1. Data Ingestion & Generation
    data_path = settings.data.raw_data_path
    if not data_path.exists():
        logger.info(f"Raw dataset not found at {data_path}. Generating synthetic dataset...")
        generator = TransactionGenerator(seed=settings.data.random_seed)
        df = generator.generate_and_save(
            output_path=data_path,
            n_samples=settings.data.sample_size,
            fraud_rate=settings.data.fraud_rate,
        )
    else:
        logger.info(f"Loading existing transactions from {data_path}...")
        df = pd.read_parquet(data_path)

    # 2. Data Validation
    validator = DataValidator()
    is_valid, val_report = validator.validate(df)
    if not is_valid:
        raise ValueError(f"Data validation failed: {val_report.errors}")

    # Display Dataset Statistics Table
    data_table = Table(title="Dataset Ingestion & Validation Summary", header_style="bold cyan")
    data_table.add_column("Metric", style="dim")
    data_table.add_column("Value")
    data_table.add_row("Total Records", f"{val_report.total_records:,}")
    data_table.add_row("Total Fraud Cases", f"{val_report.fraud_count:,}")
    data_table.add_row("Base Fraud Rate", f"{val_report.fraud_rate:.2%}")
    data_table.add_row("Time Window", f"{val_report.min_timestamp} -> {val_report.max_timestamp}")
    data_table.add_row("Duplicate IDs", str(val_report.duplicate_ids))
    data_table.add_row("Validation Status", "[green]PASSED[/green]")
    console.print(data_table)

    # 3. Temporal Splitting
    splitter = TemporalSplitter(
        train_ratio=settings.split.train_ratio,
        val_ratio=settings.split.val_ratio,
        test_ratio=settings.split.test_ratio,
    )
    splits = splitter.split(df)
    split_summary = splits.summary()

    split_table = Table(title="Temporal Splitting (Zero Lookahead)", header_style="bold magenta")
    split_table.add_column("Partition", style="bold")
    split_table.add_column("Records")
    split_table.add_column("Fraud Rate")
    split_table.add_column("Start Time")
    split_table.add_column("End Time")
    for name, stats in split_summary.items():
        split_table.add_row(
            name.upper(),
            f"{stats['records']:,}",
            f"{float(stats['fraud_rate']):.2%}",
            str(stats["start"]),
            str(stats["end"]),
        )
    console.print(split_table)

    # 4. Feature Engineering
    logger.info("Fitting and transforming leak-free feature pipelines...")
    pipeline = FeaturePipeline()
    pipeline.fit(splits.train_df)

    train_data = pipeline.extract_features(splits.train_df)
    val_data = pipeline.extract_features(splits.val_df, state_df=splits.train_df)
    test_data = pipeline.extract_features(
        splits.test_df,
        state_df=pd.concat([splits.train_df, splits.val_df], ignore_index=True),
    )

    feature_table = Table(title="Extracted Features", header_style="bold yellow")
    feature_table.add_column("#", style="dim")
    feature_table.add_column("Feature Name")
    for idx, f_name in enumerate(train_data.feature_names, start=1):
        feature_table.add_row(str(idx), f_name)
    console.print(feature_table)

    # 5. Model Training (Baselines & LightGBM)
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    Path("artifacts/models").mkdir(parents=True, exist_ok=True)

    trainer = ModelTrainer()

    # Model 1: Logistic Regression
    lr_result = trainer.train_logistic_regression(
        X_train=train_data.features,
        y_train=train_data.labels,
        X_val=val_data.features,
        y_val=val_data.labels,
        X_test=test_data.features,
        y_test=test_data.labels,
    )

    # Model 2: LightGBM
    lgb_result = trainer.train_lightgbm(
        X_train=train_data.features,
        y_train=train_data.labels,
        X_val=val_data.features,
        y_val=val_data.labels,
        X_test=test_data.features,
        y_test=test_data.labels,
    )

    # 6. Evaluation Comparison Table
    eval_table = Table(title="Model Evaluation Benchmark (Test Set)", header_style="bold green")
    eval_table.add_column("Metric", style="bold")
    eval_table.add_column("Logistic Regression (Baseline)")
    eval_table.add_column("LightGBM")
    eval_table.add_column("Improvement")

    metrics_to_compare = [
        ("PR-AUC", lr_result.test_report.pr_auc, lgb_result.test_report.pr_auc, True),
        ("ROC-AUC", lr_result.test_report.roc_auc, lgb_result.test_report.roc_auc, True),
        (
            "Brier Score",
            lr_result.test_report.brier_score,
            lgb_result.test_report.brier_score,
            False,
        ),
        ("Precision", lr_result.test_report.precision, lgb_result.test_report.precision, True),
        ("Recall", lr_result.test_report.recall, lgb_result.test_report.recall, True),
        ("F1-Score", lr_result.test_report.f1, lgb_result.test_report.f1, True),
        ("False Positive Rate", lr_result.test_report.fpr, lgb_result.test_report.fpr, False),
        ("False Negative Rate", lr_result.test_report.fnr, lgb_result.test_report.fnr, False),
        ("Optimal Threshold", lr_result.optimal_threshold, lgb_result.optimal_threshold, None),
        (
            "Expected Cost ($)",
            lr_result.test_report.expected_cost,
            lgb_result.test_report.expected_cost,
            False,
        ),
    ]

    for label, lr_val, lgb_val, higher_is_better in metrics_to_compare:
        if higher_is_better is True:
            diff = lgb_val - lr_val
            diff_str = f"[green]+{diff:.4f}[/green]" if diff >= 0 else f"[red]{diff:.4f}[/red]"
        elif higher_is_better is False:
            diff = lgb_val - lr_val
            diff_str = f"[green]{diff:.2f}[/green]" if diff <= 0 else f"[red]+{diff:.2f}[/red]"
        else:
            diff_str = "—"

        eval_table.add_row(
            label,
            f"{lr_val:.4f}" if isinstance(lr_val, float) else str(lr_val),
            f"{lgb_val:.4f}" if isinstance(lgb_val, float) else str(lgb_val),
            diff_str,
        )
    console.print(eval_table)

    # 7. Confusion Matrix Tables
    cm_table = Table(title="Confusion Matrix Breakdown (Test Set)", header_style="bold blue")
    cm_table.add_column("Model")
    cm_table.add_column("True Negatives (TN)")
    cm_table.add_column("False Positives (FP)")
    cm_table.add_column("False Negatives (FN)")
    cm_table.add_column("True Positives (TP)")
    cm_table.add_row(
        "Logistic Regression",
        str(lr_result.test_report.tn),
        f"[red]{lr_result.test_report.fp}[/red]",
        f"[red]{lr_result.test_report.fn}[/red]",
        f"[green]{lr_result.test_report.tp}[/green]",
    )
    cm_table.add_row(
        "LightGBM",
        str(lgb_result.test_report.tn),
        f"[red]{lgb_result.test_report.fp}[/red]",
        f"[red]{lgb_result.test_report.fn}[/red]",
        f"[green]{lgb_result.test_report.tp}[/green]",
    )
    console.print(cm_table)

    # 8. Save Best Model Artifact
    best_result = (
        lgb_result if lgb_result.test_report.pr_auc >= lr_result.test_report.pr_auc else lr_result
    )
    save_path = Path("artifacts/models/best_fraud_model.joblib")
    joblib.dump(
        {
            "model_name": best_result.model_name,
            "model": best_result.model,
            "calibrator": best_result.calibrator,
            "optimal_threshold": best_result.optimal_threshold,
            "feature_names": train_data.feature_names,
            "metrics": best_result.test_report.to_dict(),
        },
        save_path,
    )
    logger.info(f"Saved production-ready candidate model to {save_path}")
    logger.info("MLflow experiments logged to SQLite store: 'mlflow.db'")
    logger.info("Phase 1 Offline ML Pipeline completed successfully!")


if __name__ == "__main__":
    run_training_pipeline()
