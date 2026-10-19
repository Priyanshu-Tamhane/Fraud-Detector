"""Real-time streaming transaction consumer, scorer, and DLQ router."""

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from apps.stream_processor.producer import json_serializer
from apps.stream_processor.storage import TransactionStorage, storage
from ml.config import settings
from ml.data.schemas import TransactionRecord
from ml.inference.model_loader import ModelLoader, model_loader
from ml.inference.predictor import ScoredTransaction, TransactionPredictor, predictor
from ml.logging import logger


class FraudStreamProcessor:
    """Consumes transactions from Kafka, computes online features, scores risk, and sinks to DB & scored topics."""

    def __init__(
        self,
        bootstrap_servers: str | None = None,
        topic_raw: str
        | None = None,  # The inbound Kafka topic (transactions.raw) where payment switches publish unverified transactions.
        topic_scored: str
        | None = None,  # The outbound Kafka topic (transactions.scored) where downstream services (e.g., notification systems, blocklists) receive our predictions.
        topic_dlq: str | None = None,
        consumer_group: str
        | None = None,  # Kafka consumer group (fraudshield-scorer-group). Allows multiple worker replicas to share the load without duplicating work.
        pred: TransactionPredictor | None = None,
        db_storage: TransactionStorage | None = None,
        loader: ModelLoader | None = None,
    ) -> None:
        self.bootstrap_servers = bootstrap_servers or settings.kafka.bootstrap_servers
        self.topic_raw = topic_raw or settings.kafka.topic_raw
        self.topic_scored = topic_scored or settings.kafka.topic_scored
        self.topic_dlq = topic_dlq or settings.kafka.topic_dlq
        self.consumer_group = consumer_group or settings.kafka.consumer_group

        self.predictor = pred or predictor
        self.storage = db_storage or storage
        self.loader = loader or model_loader

        self.consumer: AIOKafkaConsumer | None = None
        self.producer: AIOKafkaProducer | None = None
        self._running: bool = False

        self.processed_count: int = 0
        self.dlq_count: int = 0

    async def start(self) -> None:
        """Starts Kafka consumer, output producer, database connection, and ensures model is loaded."""
        # 1. Ensure model is loaded in memory
        if not self.loader.is_loaded:
            self.loader.load()

        # 2. Initialize database
        await self.storage.init_db()

        # 3. Initialize output producer for scored and DLQ topics
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=json_serializer,
            key_serializer=lambda k: k.encode("utf-8") if isinstance(k, str) else k,
        )
        await self.producer.start()

        # 4. Initialize consumer
        self.consumer = AIOKafkaConsumer(
            self.topic_raw,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.consumer_group,
            auto_offset_reset=settings.kafka.auto_offset_reset,
            enable_auto_commit=True,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )
        await self.consumer.start()
        self._running = True
        logger.info(
            f"FraudStreamProcessor started. Subscribed to '{self.topic_raw}' "
            f"under consumer group '{self.consumer_group}'."
        )

    async def stop(self) -> None:
        """Stops stream processor components cleanly."""
        self._running = False
        if self.consumer:
            await self.consumer.stop()
            logger.info("Kafka consumer stopped.")
        if self.producer:
            await self.producer.stop()
            logger.info("Kafka output producer stopped.")
        await self.storage.close()

    async def process_record(self, raw_data: dict[str, Any]) -> dict[str, Any]:
        """Validates payload, scores transaction, handles DLQ routing, and persists audit record."""
        # 1. Payload validation
        try:
            # Parse timestamp if string
            ts = raw_data.get("timestamp")
            if isinstance(ts, str):
                parsed_ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            elif isinstance(ts, datetime):
                parsed_ts = ts
            else:
                parsed_ts = datetime.now(UTC)

            # Validate against schema rules
            validated = TransactionRecord(
                transaction_id=str(raw_data["transaction_id"]),
                user_id=str(raw_data["user_id"]),
                merchant_id=str(raw_data["merchant_id"]),
                device_id=str(raw_data.get("device_id", "device_unknown")),
                amount=float(raw_data["amount"]),
                currency=str(raw_data.get("currency", "INR")),
                timestamp=parsed_ts,
                ip_address=str(raw_data.get("ip_address", "127.0.0.1")),
                merchant_category=str(raw_data.get("merchant_category", "retail")),
            )
        except Exception as validation_err:
            # Route corrupted/invalid payload to Dead Letter Queue (DLQ)
            self.dlq_count += 1
            dlq_payload = {
                "error": str(validation_err),
                "original_payload": raw_data,
                "failed_at": datetime.now(UTC).isoformat(),
            }
            logger.warning(f"Routing invalid transaction to DLQ: {validation_err}")
            if self.producer:
                await self.producer.send_and_wait(
                    topic=self.topic_dlq,
                    key=str(raw_data.get("transaction_id", "unknown_tx")),
                    value=dlq_payload,
                )
            return {"status": "DLQ", "reason": str(validation_err)}

        # 2. Run real-time feature computation & inference
        scored: ScoredTransaction = self.predictor.predict(
            transaction_id=validated.transaction_id,
            user_id=validated.user_id,
            merchant_id=validated.merchant_id,
            device_id=validated.device_id,
            amount=validated.amount,
            timestamp=validated.timestamp,
            merchant_category=validated.merchant_category,
        )

        result_payload = {
            "transaction_id": scored.transaction_id,
            "user_id": validated.user_id,
            "merchant_id": validated.merchant_id,
            "amount": validated.amount,
            "currency": validated.currency,
            "timestamp": validated.timestamp.isoformat(),
            "fraud_probability": scored.fraud_probability,
            "risk_score": scored.risk_score,
            "decision": str(scored.decision),
            "top_risk_factors": scored.top_risk_factors,
            "model_version": scored.model_version,
            "latency_ms": scored.latency_ms,
            "scored_at": datetime.now(UTC).isoformat(),
        }

        # 3. Publish to transactions.scored topic
        if self.producer:
            await self.producer.send_and_wait(
                topic=self.topic_scored,
                key=validated.user_id,
                value=result_payload,
            )

        # 4. Asynchronously persist to database audit sink
        await self.storage.save_transaction(
            scored=scored,
            user_id=validated.user_id,
            merchant_id=validated.merchant_id,
            amount=validated.amount,
            currency=validated.currency,
            timestamp=validated.timestamp,
        )

        self.processed_count += 1
        return result_payload

    async def run(self, max_messages: int | None = None) -> None:
        """Main event processing loop."""
        if not self.consumer:
            raise RuntimeError("Consumer is not started.")

        logger.info(f"Polling topic '{self.topic_raw}' for transactions...")
        msg_count = 0

        async for msg in self.consumer:
            if not self._running:
                break

            await self.process_record(msg.value)
            msg_count += 1

            if max_messages and msg_count >= max_messages:
                logger.info(f"Reached max_messages limit ({max_messages}). Halting consumer.")
                break


# Global stream processor instance
stream_processor = FraudStreamProcessor()


if __name__ == "__main__":

    async def main() -> None:
        await stream_processor.start()
        try:
            await stream_processor.run()
        except KeyboardInterrupt:
            logger.info("Interrupted. Shutting down stream processor...")
        finally:
            await stream_processor.stop()

    asyncio.run(main())
