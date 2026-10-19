"""Redis-backed online feature store with sliding velocity windows and in-memory fallback."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime

import redis.asyncio as aioredis

from ml.config import settings
from ml.logging import logger


@dataclass(frozen=True)
class OnlineFeatures:
    """Calculated real-time behavioral and velocity feature values."""

    user_prior_count: int
    user_prior_mean_amount: float
    amount_to_user_mean_ratio: float
    is_new_device_for_user: int
    device_unique_users_seen: int
    tx_count_5m: int
    tx_count_1h: int
    tx_count_24h: int
    amount_sum_24h: float


class OnlineFeatureStore:
    """Computes and caches online behavioral & sliding-window velocity features using Redis."""

    def __init__(
        self,
        redis_url: str | None = None,
        ttl_user_window: int | None = None,
        ttl_device_set: int | None = None,
    ) -> None:
        self.redis_url = redis_url or settings.redis.redis_url
        self.ttl_user_window = ttl_user_window or settings.redis.ttl_user_window_seconds
        self.ttl_device_set = ttl_device_set or settings.redis.ttl_device_set_seconds

        self._client: aioredis.Redis | None = None
        self._using_fallback: bool = False

        # In-memory sliding-window fallback for local development & isolated tests
        self._fallback_user_txs: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
        self._fallback_user_cumulative: dict[str, tuple[int, float]] = defaultdict(lambda: (0, 0.0))
        self._fallback_user_devices: dict[str, set[str]] = defaultdict(set)
        self._fallback_device_users: dict[str, set[str]] = defaultdict(set)

    async def connect(self) -> None:
        """Connects to Redis; switches to in-memory fallback if unreachable."""
        try:
            client = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=1.5,
            )
            await client.ping()
            self._client = client
            self._using_fallback = False
            logger.info(f"Connected to Redis feature store at '{self.redis_url}'.")
        except Exception as e:
            logger.warning(
                f"Redis unavailable ({e}). Falling back to local in-memory sliding-window state."
            )
            self._using_fallback = True
            self._client = None

    async def close(self) -> None:
        """Closes Redis client connections."""
        if self._client:
            await self._client.aclose()
            logger.info("Closed Redis connection.")

    async def get_and_update_features(
        self,
        user_id: str,
        device_id: str,
        amount: float,
        timestamp: datetime,
        tx_id: str = "tx",
        update_state: bool = True,
    ) -> OnlineFeatures:
        """Extracts features strictly prior to current transaction, then updates state."""
        # Convert timestamp to epoch seconds
        if timestamp.tzinfo is None:
            ts_epoch = timestamp.replace(tzinfo=UTC).timestamp()
        else:
            ts_epoch = timestamp.timestamp()

        if self._using_fallback or not self._client:
            return self._compute_fallback(
                user_id=user_id,
                device_id=device_id,
                amount=amount,
                ts_epoch=ts_epoch,
                tx_id=tx_id,
                update_state=update_state,
            )
        else:
            return await self._compute_redis(
                user_id=user_id,
                device_id=device_id,
                amount=amount,
                ts_epoch=ts_epoch,
                tx_id=tx_id,
                update_state=update_state,
            )

    def get_and_update_features_sync(
        self,
        user_id: str,
        device_id: str,
        amount: float,
        timestamp: datetime,
        tx_id: str = "tx",
        update_state: bool = True,
    ) -> OnlineFeatures:
        """Synchronously extracts online features using the in-memory fallback state."""
        if timestamp.tzinfo is None:
            ts_epoch = timestamp.replace(tzinfo=UTC).timestamp()
        else:
            ts_epoch = timestamp.timestamp()

        return self._compute_fallback(
            user_id=user_id,
            device_id=device_id,
            amount=amount,
            ts_epoch=ts_epoch,
            tx_id=tx_id,
            update_state=update_state,
        )

    # -------------------------------------------------------------------------
    # Redis Implementation (Sliding Window ZSET + SETs with Pipelines)
    # -------------------------------------------------------------------------
    async def _compute_redis(
        self,
        user_id: str,
        device_id: str,
        amount: float,
        ts_epoch: float,
        tx_id: str,
        update_state: bool,
    ) -> OnlineFeatures:
        assert self._client is not None
        pipe = self._client.pipeline()

        user_tx_key = f"user:{user_id}:txs"
        user_cum_key = f"user:{user_id}:cumulative"
        user_devices_key = f"user:{user_id}:devices"
        device_users_key = f"device:{device_id}:users"

        # 1. Query prior state via pipelining (1 round-trip)
        pipe.hgetall(user_cum_key)
        pipe.sismember(user_devices_key, device_id)
        pipe.sismember(device_users_key, user_id)
        pipe.scard(device_users_key)

        # Sliding window range queries
        pipe.zrangebyscore(user_tx_key, min=ts_epoch - 300, max=ts_epoch)  # 5 min
        pipe.zrangebyscore(user_tx_key, min=ts_epoch - 3600, max=ts_epoch)  # 1 hour
        pipe.zrangebyscore(user_tx_key, min=ts_epoch - 86400, max=ts_epoch)  # 24 hours

        results = await pipe.execute()

        cum_data = results[0] or {}
        is_device_member = bool(results[1])
        is_user_on_device = bool(results[2])
        raw_device_users = int(results[3])
        device_unique_users = raw_device_users if is_user_on_device else (raw_device_users + 1)

        txs_5m = results[4]
        txs_1h = results[5]
        txs_24h = results[6]

        prior_count = int(cum_data.get("count", 0))
        prior_sum = float(cum_data.get("sum", 0.0))
        prior_mean = (prior_sum / prior_count) if prior_count > 0 else 0.0
        amount_ratio = amount / max(prior_mean, 10.0)

        is_new_device = 0 if is_device_member else 1

        # Sum amounts in 24h window
        amount_sum_24h = 0.0
        for entry in txs_24h:
            parts = entry.split(":")
            if len(parts) >= 2:
                amount_sum_24h += float(parts[1])

        features = OnlineFeatures(
            user_prior_count=prior_count,
            user_prior_mean_amount=prior_mean,
            amount_to_user_mean_ratio=amount_ratio,
            is_new_device_for_user=is_new_device,
            device_unique_users_seen=max(device_unique_users, 1),
            tx_count_5m=len(txs_5m),
            tx_count_1h=len(txs_1h),
            tx_count_24h=len(txs_24h),
            amount_sum_24h=amount_sum_24h,
        )

        # 2. Update state if requested (atomic pipeline)
        if update_state:
            up_pipe = self._client.pipeline()
            # Cumulative tracking
            up_pipe.hincrby(user_cum_key, "count", 1)
            up_pipe.hincrbyfloat(user_cum_key, "sum", amount)

            # Add to sliding window sorted set
            member_val = f"{ts_epoch}:{amount}:{tx_id}"
            up_pipe.zadd(user_tx_key, {member_val: ts_epoch})
            up_pipe.expire(user_tx_key, self.ttl_user_window)

            # Clean expired records older than 24h
            up_pipe.zremrangebyscore(user_tx_key, min="-inf", max=ts_epoch - 86400)

            # Device tracking
            up_pipe.sadd(user_devices_key, device_id)
            up_pipe.expire(user_devices_key, self.ttl_device_set)
            up_pipe.sadd(device_users_key, user_id)
            up_pipe.expire(device_users_key, self.ttl_device_set)

            await up_pipe.execute()

        return features

    # -------------------------------------------------------------------------
    # In-Memory Fallback Implementation (Guarantees Offline Test Independence)
    # -------------------------------------------------------------------------
    def _compute_fallback(
        self,
        user_id: str,
        device_id: str,
        amount: float,
        ts_epoch: float,
        tx_id: str,
        update_state: bool,
    ) -> OnlineFeatures:
        # Prior cumulative stats
        prior_count, prior_sum = self._fallback_user_cumulative[user_id]
        prior_mean = (prior_sum / prior_count) if prior_count > 0 else 0.0
        amount_ratio = amount / max(prior_mean, 10.0)

        is_new_device = 1 if device_id not in self._fallback_user_devices[user_id] else 0
        known_device_users = self._fallback_device_users[device_id]
        device_users_count = len(known_device_users | {user_id})

        # Sliding window velocity calculations
        history = self._fallback_user_txs[user_id]
        txs_5m = [tx for tx in history if (ts_epoch - 300) <= tx[0] <= ts_epoch]
        txs_1h = [tx for tx in history if (ts_epoch - 3600) <= tx[0] <= ts_epoch]
        txs_24h = [tx for tx in history if (ts_epoch - 86400) <= tx[0] <= ts_epoch]
        amount_sum_24h = sum(tx[1] for tx in txs_24h)

        features = OnlineFeatures(
            user_prior_count=prior_count,
            user_prior_mean_amount=prior_mean,
            amount_to_user_mean_ratio=amount_ratio,
            is_new_device_for_user=is_new_device,
            device_unique_users_seen=max(device_users_count, 1),
            tx_count_5m=len(txs_5m),
            tx_count_1h=len(txs_1h),
            tx_count_24h=len(txs_24h),
            amount_sum_24h=amount_sum_24h,
        )

        if update_state:
            self._fallback_user_cumulative[user_id] = (prior_count + 1, prior_sum + amount)
            self._fallback_user_txs[user_id].append((ts_epoch, amount, tx_id))
            self._fallback_user_devices[user_id].add(device_id)
            self._fallback_device_users[device_id].add(user_id)

        return features

    def clear(self) -> None:
        """Clears in-memory fallback state (used in unit test teardowns)."""
        self._fallback_user_txs.clear()
        self._fallback_user_cumulative.clear()
        self._fallback_user_devices.clear()
        self._fallback_device_users.clear()


# Global feature store instance
online_feature_store = OnlineFeatureStore()
