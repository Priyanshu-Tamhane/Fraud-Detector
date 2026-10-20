"""Pipeline command to build Parquet batch tables, apply Feast registry, and materialize online store."""

from datetime import UTC, datetime, timedelta

from feature_store.client import FeatureStoreClient
from feature_store.dataset_builder import build_and_save_feature_tables
from ml.logging import logger


def run_materialization(
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> None:
    """Executes full Feast offline generation and online store materialization."""
    logger.info("Step 1/3: Building and saving batch Parquet feature tables...")
    user_pq, device_pq = build_and_save_feature_tables()
    logger.info(f"Parquet tables ready: '{user_pq}' and '{device_pq}'.")

    logger.info("Step 2/3: Applying declarative feature definitions to Feast registry...")
    client = FeatureStoreClient()
    client.apply()

    logger.info("Step 3/3: Materializing features into online store...")
    if end_date is None:
        end_date = datetime.now(tz=UTC) + timedelta(minutes=5)
    if start_date is None:
        # Default window: materialize past 90 days
        start_date = end_date - timedelta(days=90)

    client.materialize(start_date=start_date, end_date=end_date)
    logger.info("Feast materialization pipeline completed successfully!")


if __name__ == "__main__":
    run_materialization()
