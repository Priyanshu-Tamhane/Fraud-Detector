"""Audit persistence storage for scored transactions supporting PostgreSQL and SQLite fallback."""

import asyncio
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import asyncpg

from ml.config import settings
from ml.inference.predictor import ScoredTransaction
from ml.logging import logger

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS scored_transactions (
    transaction_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    merchant_id TEXT NOT NULL,
    amount REAL NOT NULL,
    currency TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    fraud_probability REAL NOT NULL,
    risk_score REAL NOT NULL,
    decision TEXT NOT NULL,
    top_risk_factors TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    model_version TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scored_user_id ON scored_transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_scored_decision ON scored_transactions(decision);
"""


class TransactionStorage:
    """Handles asynchronous persistence of scored transactions to PostgreSQL with SQLite fallback."""

    def __init__(
        self,
        postgres_url: str | None = None,
        sqlite_path: Path | None = None,
    ) -> None:
        self.postgres_url = postgres_url or settings.database.postgres_url
        self.sqlite_path = sqlite_path or settings.database.sqlite_fallback_path
        self._pg_pool: asyncpg.Pool | None = None
        self._using_sqlite: bool = False

    async def init_db(self) -> None:
        """Initializes database schema, attempting PostgreSQL first with SQLite fallback."""
        try:
            # Try connecting to PostgreSQL with short timeout
            self._pg_pool = await asyncpg.create_pool(
                self.postgres_url,
                min_size=2,
                max_size=10,
                timeout=2.0,
            )
            async with self._pg_pool.acquire() as conn:
                await conn.execute(SCHEMA_SQL)
            self._using_sqlite = False
            logger.info("Connected to PostgreSQL audit database successfully.")
        except Exception as e:
            logger.warning(
                f"PostgreSQL unavailable ({e}). Falling back to local SQLite at '{self.sqlite_path}'."
            )
            self._using_sqlite = True
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            # Initialize SQLite schema
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._init_sqlite_schema)

    def _init_sqlite_schema(self) -> None:
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()

    async def save_transaction(
        self,
        scored: ScoredTransaction,
        user_id: str,
        merchant_id: str,
        amount: float,
        currency: str = "INR",
        timestamp: str | datetime | None = None,
    ) -> None:
        """Persists a scored transaction record."""
        ts_str = (
            timestamp.isoformat()
            if isinstance(timestamp, datetime)
            else (str(timestamp) if timestamp else datetime.now(UTC).isoformat())
        )
        created_at = datetime.now(UTC).isoformat()
        factors_json = json.dumps(scored.top_risk_factors)

        record_data = (
            scored.transaction_id,
            user_id,
            merchant_id,
            float(amount),
            currency,
            ts_str,
            float(scored.fraud_probability),
            float(scored.risk_score),
            str(scored.decision),
            factors_json,
            float(scored.latency_ms),
            scored.model_version,
            created_at,
        )

        if not self._using_sqlite and self._pg_pool:
            query = """
            INSERT INTO scored_transactions (
                transaction_id, user_id, merchant_id, amount, currency, timestamp,
                fraud_probability, risk_score, decision, top_risk_factors,
                latency_ms, model_version, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            ON CONFLICT (transaction_id) DO NOTHING;
            """
            async with self._pg_pool.acquire() as conn:
                await conn.execute(query, *record_data)
        else:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._save_sqlite, record_data)

    def _save_sqlite(self, record_data: tuple[Any, ...]) -> None:
        query = """
        INSERT OR IGNORE INTO scored_transactions (
            transaction_id, user_id, merchant_id, amount, currency, timestamp,
            fraud_probability, risk_score, decision, top_risk_factors,
            latency_ms, model_version, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(query, record_data)
            conn.commit()

    async def get_transaction(self, transaction_id: str) -> dict[str, Any] | None:
        """Retrieves a scored transaction by its ID."""
        if not self._using_sqlite and self._pg_pool:
            query = "SELECT * FROM scored_transactions WHERE transaction_id = $1"
            async with self._pg_pool.acquire() as conn:
                row = await conn.fetchrow(query, transaction_id)
                if row:
                    res = dict(row)
                    res["top_risk_factors"] = json.loads(res["top_risk_factors"])
                    return res
                return None
        else:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._get_sqlite, transaction_id)

    def _get_sqlite(self, transaction_id: str) -> dict[str, Any] | None:
        query = "SELECT * FROM scored_transactions WHERE transaction_id = ?"
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, (transaction_id,))
            row = cursor.fetchone()
            if row:
                res = dict(row)
                res["top_risk_factors"] = json.loads(res["top_risk_factors"])
                return res
            return None

    async def count_transactions(self) -> int:
        """Returns total count of scored transactions."""
        if not self._using_sqlite and self._pg_pool:
            query = "SELECT COUNT(*) FROM scored_transactions"
            async with self._pg_pool.acquire() as conn:
                return await conn.fetchval(query)
        else:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._count_sqlite)

    def _count_sqlite(self) -> int:
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM scored_transactions")
            return int(cursor.fetchone()[0])

    async def close(self) -> None:
        """Closes connection pools cleanly."""
        if self._pg_pool:
            await self._pg_pool.close()
            logger.info("Closed PostgreSQL connection pool.")


# Global storage instance
storage = TransactionStorage()
