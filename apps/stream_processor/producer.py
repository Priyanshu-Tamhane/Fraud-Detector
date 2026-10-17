"""Asynchronous Kafka / Redpanda transaction stream producer."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from aiokafka import AIOKafkaProducer

from ml.config import settings
from ml.logging import logger


def json_serializer(obj: Any) -> bytes:
    """Serializes objects to JSON bytes handling datetimes."""
    if isinstance(obj, datetime):
        return obj.isoformat().encode("utf-8")
    return json.dumps(obj, default=str).encode("utf-8")


class TransactionStreamProducer:
    """Streams transactions to Kafka / Redpanda topics keyed by user_id for strict partition ordering."""

    def __init__(
        self,
        bootstrap_servers: str | None = None,
        topic: str | None = None,
    ) -> None:
        self.bootstrap_servers = bootstrap_servers or settings.kafka.bootstrap_servers
        self.topic = topic or settings.kafka.topic_raw
        self.producer: AIOKafkaProducer | None = None

    async def start(self) -> None:
        """Starts the Kafka producer client."""
        logger.info(f"Connecting stream producer to Kafka at '{self.bootstrap_servers}'...")
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=json_serializer,
            key_serializer=lambda k: k.encode("utf-8") if isinstance(k, str) else k,
            acks="all",
            retry_backoff_ms=200,
        )
        await self.producer.start()
        logger.info("Transaction stream producer connected and started.")

    async def stop(self) -> None:
        """Stops producer client."""
        if self.producer:
            await self.producer.stop()
            logger.info("Transaction stream producer stopped.")

    async def send_transaction(self, tx: dict[str, Any], topic: str | None = None) -> None:
        """Publishes a transaction to Kafka, partitioned by user_id."""
        if not self.producer:
            raise RuntimeError("Producer is not started. Call 'start()' first.")

        target_topic = topic or self.topic
        user_key = str(tx.get("user_id", "default_user"))

        await self.producer.send_and_wait(
            topic=target_topic,
            key=user_key,
            value=tx,
        )

    async def stream_from_parquet(
        self,
        parquet_path: Path | None = None,
        limit: int = 100,
        delay_seconds: float = 0.01,
    ) -> int:
        """Streams transactions from Parquet file in chronological order."""
        path = parquet_path or settings.data.raw_data_path
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found at '{path}'")

        df = pd.read_parquet(path)
        df.sort_values(by="timestamp", inplace=True)
        subset = df.head(limit)

        logger.info(
            f"Streaming {len(subset):,} transactions from '{path}' to topic '{self.topic}'..."
        )
        sent_count = 0

        for record in subset.to_dict(orient="records"):
            await self.send_transaction(record)
            sent_count += 1
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)

        logger.info(f"Successfully streamed {sent_count:,} transactions to '{self.topic}'.")
        return sent_count
