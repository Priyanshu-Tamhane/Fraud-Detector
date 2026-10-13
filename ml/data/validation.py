"""Data validation module for transaction datasets."""

from dataclasses import dataclass

import pandas as pd

from ml.logging import logger


@dataclass(frozen=True)
class ValidationReport:
    """Immutable validation summary report."""

    is_valid: bool
    total_records: int
    null_counts: dict[str, int]
    negative_amounts: int
    duplicate_ids: int
    fraud_count: int
    fraud_rate: float
    min_timestamp: str
    max_timestamp: str
    errors: list[str]


class DataValidator:
    """Validates transaction dataframes against schema, domain rules, and integrity constraints."""

    REQUIRED_COLUMNS: list[str] = [
        "transaction_id",
        "user_id",
        "merchant_id",
        "device_id",
        "amount",
        "timestamp",
        "merchant_category",
        "is_fraud",
    ]

    def validate(self, df: pd.DataFrame) -> tuple[bool, ValidationReport]:
        """Runs validation checks on the input DataFrame.

        Returns:
            Tuple of (is_valid: bool, report: ValidationReport)
        """
        errors: list[str] = []

        # 1. Check required columns
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")

        if not errors and df.empty:
            errors.append("DataFrame is empty.")

        if errors:
            report = ValidationReport(
                is_valid=False,
                total_records=len(df),
                null_counts={},
                negative_amounts=0,
                duplicate_ids=0,
                fraud_count=0,
                fraud_rate=0.0,
                min_timestamp="N/A",
                max_timestamp="N/A",
                errors=errors,
            )
            return False, report

        # 2. Check null counts
        null_counts = {
            col: int(df[col].isnull().sum()) for col in self.REQUIRED_COLUMNS if col in df.columns
        }
        critical_nulls = {
            col: cnt
            for col, cnt in null_counts.items()
            if cnt > 0 and col in ["transaction_id", "user_id", "amount", "timestamp"]
        }
        if critical_nulls:
            errors.append(f"Critical columns contain nulls: {critical_nulls}")

        # 3. Check negative or zero amounts
        invalid_amounts = int((df["amount"] <= 0).sum())
        if invalid_amounts > 0:
            errors.append(f"Found {invalid_amounts} records with amount <= 0")

        # 4. Check duplicate transaction IDs
        duplicate_ids = int(df["transaction_id"].duplicated().sum())
        if duplicate_ids > 0:
            errors.append(f"Found {duplicate_ids} duplicate transaction IDs")

        # 5. Check labels
        invalid_labels = int((~df["is_fraud"].isin([0, 1])).sum())
        if invalid_labels > 0:
            errors.append(f"Found {invalid_labels} labels that are not 0 or 1")

        fraud_count = int(df["is_fraud"].sum())
        fraud_rate = float(fraud_count / len(df)) if len(df) > 0 else 0.0

        ts_series = pd.to_datetime(df["timestamp"])
        min_ts = ts_series.min().isoformat()
        max_ts = ts_series.max().isoformat()

        is_valid = len(errors) == 0

        report = ValidationReport(
            is_valid=is_valid,
            total_records=len(df),
            null_counts=null_counts,
            negative_amounts=invalid_amounts,
            duplicate_ids=duplicate_ids,
            fraud_count=fraud_count,
            fraud_rate=fraud_rate,
            min_timestamp=min_ts,
            max_timestamp=max_ts,
            errors=errors,
        )

        if not is_valid:
            logger.error(f"Validation failed with {len(errors)} error(s): {errors}")
        else:
            logger.info(
                f"Validation passed: {report.total_records:,} records | "
                f"Fraud rate: {report.fraud_rate:.2%} | Time range: {min_ts} -> {max_ts}"
            )

        return is_valid, report
