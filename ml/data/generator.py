"""Synthetic transaction generator simulating realistic temporal fraud dynamics."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ml.logging import logger


class TransactionGenerator:
    """Generates synthetic financial transactions with realistic legitimate and fraudulent behavioral patterns."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.categories: list[str] = [
            "retail",
            "grocery",
            "electronics",
            "travel",
            "entertainment",
            "dining",
            "crypto_exchange",
        ]
        self.cities: list[str] = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Pune", "Chennai"]

    def generate(
        self,
        n_samples: int = 50_000,
        n_users: int = 2_000,
        n_merchants: int = 500,
        n_devices: int = 3_000,
        fraud_rate: float = 0.03,
        start_time: datetime | None = None,
    ) -> pd.DataFrame:
        """Generates a DataFrame of synthetic transactions strictly ordered by timestamp."""
        if start_time is None:
            start_time = datetime(2026, 1, 1, 0, 0, 0)

        logger.info(
            f"Generating {n_samples:,} synthetic transactions (target fraud rate: {fraud_rate:.1%})..."
        )

        # Generate user profiles (baseline mean log-amount, preferred device)
        user_ids = [f"user_{i:04d}" for i in range(n_users)]
        merchant_ids = [f"merchant_{i:04d}" for i in range(n_merchants)]
        device_ids = [f"device_{i:05d}" for i in range(n_devices)]

        user_mean_amounts = {u: float(self.rng.uniform(300.0, 5000.0)) for u in user_ids}
        user_primary_device = {u: self.rng.choice(device_ids) for u in user_ids}

        records: list[dict] = []
        current_time = start_time

        # Create time progression: random inter-arrival intervals
        inter_arrival_seconds = self.rng.exponential(scale=30.0, size=n_samples)

        for i in range(n_samples):
            current_time += timedelta(seconds=float(inter_arrival_seconds[i]))
            user = self.rng.choice(user_ids)
            merchant = self.rng.choice(merchant_ids)
            category = self.rng.choice(
                self.categories, p=[0.30, 0.25, 0.15, 0.08, 0.10, 0.10, 0.02]
            )
            city = self.rng.choice(self.cities)

            # Determine if this transaction will be fraud
            is_fraud = int(self.rng.random() < fraud_rate)

            if is_fraud:
                # Fraud pattern generation:
                fraud_mode = self.rng.choice(["burst", "spike", "new_device", "night_anomaly"])
                if fraud_mode == "spike":
                    # Anomaly: 10x - 25x user normal spending in high-risk category
                    amount = round(float(user_mean_amounts[user] * self.rng.uniform(10.0, 25.0)), 2)
                    device = self.rng.choice(device_ids)
                    category = "electronics" if self.rng.random() < 0.6 else "crypto_exchange"
                elif fraud_mode == "burst":
                    # Velocity attack: quick transaction
                    amount = round(float(user_mean_amounts[user] * self.rng.uniform(2.0, 5.0)), 2)
                    device = self.rng.choice(device_ids)
                elif fraud_mode == "new_device":
                    # Brand new unassociated device
                    amount = round(float(user_mean_amounts[user] * self.rng.uniform(1.5, 4.0)), 2)
                    device = f"device_fraud_{self.rng.integers(1000, 9999)}"
                else:  # night_anomaly
                    amount = round(float(user_mean_amounts[user] * self.rng.uniform(3.0, 8.0)), 2)
                    device = user_primary_device[user]
            else:
                # Legitimate transaction: log-normal around user baseline
                baseline = user_mean_amounts[user]
                amount = round(float(self.rng.lognormal(mean=np.log(baseline), sigma=0.4)), 2)
                # 90% chance using primary device
                device = (
                    user_primary_device[user]
                    if self.rng.random() < 0.90
                    else self.rng.choice(device_ids)
                )

            # Safeguard amount > 0
            amount = max(amount, 10.0)

            records.append(
                {
                    "transaction_id": f"tx_{i:07d}",
                    "user_id": user,
                    "merchant_id": merchant,
                    "device_id": device,
                    "amount": amount,
                    "currency": "INR",
                    "timestamp": current_time,
                    "ip_address": f"192.168.{self.rng.integers(1, 255)}.{self.rng.integers(1, 255)}",
                    "merchant_category": category,
                    "country": "IN",
                    "city": city,
                    "is_fraud": is_fraud,
                }
            )

        df = pd.DataFrame(records)
        df.sort_values(by="timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        logger.info(
            f"Successfully generated {len(df):,} records | "
            f"Observed fraud rate: {df['is_fraud'].mean():.2%} | "
            f"Time span: {df['timestamp'].min()} to {df['timestamp'].max()}"
        )
        return df

    def generate_and_save(
        self, output_path: Path, n_samples: int = 50_000, fraud_rate: float = 0.03
    ) -> pd.DataFrame:
        """Generates transactions and saves them to a parquet file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = self.generate(n_samples=n_samples, fraud_rate=fraud_rate)
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved dataset to {output_path}")
        return df
