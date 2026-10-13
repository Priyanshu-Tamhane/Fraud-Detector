"""Temporal dataset splitting strictly preventing time-travel leakage."""

from dataclasses import dataclass

import pandas as pd

from ml.logging import logger


@dataclass(frozen=True)
class TemporalSplits:
    """Immutable container holding train, validation, and test datasets."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame

    def summary(self) -> dict[str, dict[str, str | int | float]]:
        """Provides summary metrics of the temporal partitions."""
        return {
            "train": {
                "records": len(self.train_df),
                "fraud_count": int(self.train_df["is_fraud"].sum()),
                "fraud_rate": float(self.train_df["is_fraud"].mean()),
                "start": str(self.train_df["timestamp"].min()),
                "end": str(self.train_df["timestamp"].max()),
            },
            "validation": {
                "records": len(self.val_df),
                "fraud_count": int(self.val_df["is_fraud"].sum()),
                "fraud_rate": float(self.val_df["is_fraud"].mean()),
                "start": str(self.val_df["timestamp"].min()),
                "end": str(self.val_df["timestamp"].max()),
            },
            "test": {
                "records": len(self.test_df),
                "fraud_count": int(self.test_df["is_fraud"].sum()),
                "fraud_rate": float(self.test_df["is_fraud"].mean()),
                "start": str(self.test_df["timestamp"].min()),
                "end": str(self.test_df["timestamp"].max()),
            },
        }


class TemporalSplitter:
    """Performs strict chronological splitting of transaction time-series data."""

    def __init__(
        self, train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15
    ) -> None:
        total = round(train_ratio + val_ratio + test_ratio, 6)
        if total != 1.0:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split(self, df: pd.DataFrame) -> TemporalSplits:
        """Splits DataFrame strictly by timestamp order with zero temporal overlap."""
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must contain a 'timestamp' column for temporal splitting.")

        # Ensure sorted
        sorted_df = df.sort_values(by="timestamp").reset_index(drop=True)
        n = len(sorted_df)

        train_end_idx = int(n * self.train_ratio)
        val_end_idx = train_end_idx + int(n * self.val_ratio)

        train_df = sorted_df.iloc[:train_end_idx].copy().reset_index(drop=True)
        val_df = sorted_df.iloc[train_end_idx:val_end_idx].copy().reset_index(drop=True)
        test_df = sorted_df.iloc[val_end_idx:].copy().reset_index(drop=True)

        # Strict temporal validation assertions
        max_train_ts = train_df["timestamp"].max()
        min_val_ts = val_df["timestamp"].min()
        max_val_ts = val_df["timestamp"].max()
        min_test_ts = test_df["timestamp"].min()

        if max_train_ts > min_val_ts:
            raise ValueError(
                f"Data leakage detected! Train max timestamp ({max_train_ts}) "
                f"> Validation min timestamp ({min_val_ts})"
            )
        if max_val_ts > min_test_ts:
            raise ValueError(
                f"Data leakage detected! Validation max timestamp ({max_val_ts}) "
                f"> Test min timestamp ({min_test_ts})"
            )

        splits = TemporalSplits(train_df=train_df, val_df=val_df, test_df=test_df)
        logger.info(
            f"Temporal split complete: Train={len(train_df):,} | Val={len(val_df):,} | Test={len(test_df):,}"
        )
        return splits
