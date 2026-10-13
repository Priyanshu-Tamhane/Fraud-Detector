"""End-to-end integration test for training and evaluation pipeline."""

from ml.data.generator import TransactionGenerator
from ml.data.validation import DataValidator
from ml.evaluation.metrics import MetricsCalculator
from ml.features.pipeline import FeaturePipeline
from ml.features.temporal_split import TemporalSplitter
from ml.training.trainer import ModelTrainer


def test_end_to_end_training_flow() -> None:
    # 1. Generate small sample
    generator = TransactionGenerator(seed=123)
    df = generator.generate(
        n_samples=500, n_users=50, n_merchants=20, n_devices=60, fraud_rate=0.08
    )

    # 2. Validate
    validator = DataValidator()
    is_valid, _ = validator.validate(df)
    assert is_valid is True

    # 3. Split
    splitter = TemporalSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    splits = splitter.split(df)

    # 4. Features
    pipeline = FeaturePipeline()
    pipeline.fit(splits.train_df)
    train_data = pipeline.extract_features(splits.train_df)
    val_data = pipeline.extract_features(splits.val_df, state_df=splits.train_df)
    test_data = pipeline.extract_features(splits.test_df, state_df=splits.train_df)

    # 5. Train
    metrics_calc = MetricsCalculator(cost_fn=100.0, cost_fp=5.0)
    trainer = ModelTrainer(metrics_calc=metrics_calc)

    res_lr = trainer.train_logistic_regression(
        X_train=train_data.features,
        y_train=train_data.labels,
        X_val=val_data.features,
        y_val=val_data.labels,
        X_test=test_data.features,
        y_test=test_data.labels,
    )
    assert res_lr.test_report.roc_auc >= 0.5
    assert 0.0 <= res_lr.test_report.brier_score <= 1.0

    res_lgb = trainer.train_lightgbm(
        X_train=train_data.features,
        y_train=train_data.labels,
        X_val=val_data.features,
        y_val=val_data.labels,
        X_test=test_data.features,
        y_test=test_data.labels,
    )
    assert res_lgb.test_report.roc_auc >= 0.5
    assert res_lgb.optimal_threshold > 0.0
