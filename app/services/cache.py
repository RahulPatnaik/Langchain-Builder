import json
from typing import Any, Optional
import aioredis
from loguru import logger

class RedisCacheService:
    """
    A simple service wrapper around aioredis for basic caching operations.
    """
    def __init__(self, redis_client: aioredis.Redis, default_ttl: int = 300):
        """
        Initializes the cache service.

        Args:
            redis_client: An initialized aioredis.Redis client instance (from pool).
            default_ttl: Default time-to-live for cache entries in seconds.
        """
        self.redis = redis_client
        self.default_ttl = default_ttl
        logger.info(f"RedisCacheService initialized with default TTL: {default_ttl}s")

    async def get(self, key: str) -> Optional[Any]:
        """
        Retrieves an item from the cache.

        Args:
            key: The cache key.

        Returns:
            The deserialized cached value, or None if not found or error occurs.
        """
        if not self.redis:
            logger.warning("Attempted cache GET when Redis client is unavailable.")
            return None
        try:
            cached_value = await self.redis.get(key)
            if cached_value:
                logger.debug(f"Cache HIT for key: {key}")
                # Deserialize from JSON
                return json.loads(cached_value)
            else:
                logger.debug(f"Cache MISS for key: {key}")
                return None
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from cache for key: {key}. Value might be corrupted.")
            # Optionally delete the corrupted key: await self.delete(key)
            return None
        except aioredis.RedisError as e:
            logger.error(f"Redis GET error for key {key}: {e}")
            return None
        except Exception as e:
             logger.error(f"Unexpected error during cache GET for key {key}: {e}")
             return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Stores an item in the cache.

        Args:
            key: The cache key.
            value: The value to store (must be JSON serializable).
            ttl: Optional time-to-live in seconds (overrides default).

        Returns:
            True if the item was successfully set, False otherwise.
        """
        if not self.redis:
            logger.warning(f"Attempted cache SET for key '{key}' when Redis client is unavailable.")
            return False

        expire_time = ttl if ttl is not None else self.default_ttl
        try:
            # Serialize value to JSON string
            serialized_value = json.dumps(value)
            result = await self.redis.set(key, serialized_value, ex=expire_time)
            if result:
                logger.debug(f"Cache SET successful for key: {key} with TTL: {expire_time}s")
                return True
            else:
                # Should not happen with basic SET unless Redis command fails unexpectedly
                logger.warning(f"Redis SET command returned non-True for key: {key}")
                return False
        except TypeError as e:
            logger.error(f"Failed to serialize value to JSON for cache key {key}: {e}")
            return False
        except aioredis.RedisError as e:
            logger.error(f"Redis SET error for key {key}: {e}")
            return False
        except Exception as e:
             logger.error(f"Unexpected error during cache SET for key {key}: {e}")
             return False

    async def delete(self, key: str) -> bool:
        """
        Deletes an item from the cache.

        Args:
            key: The cache key to delete.

        Returns:
            True if the key was deleted or didn't exist, False on error.
        """
        if not self.redis:
            logger.warning(f"Attempted cache DELETE for key '{key}' when Redis client is unavailable.")
            return False
        try:
            result = await self.redis.delete(key)
            # delete returns the number of keys deleted (0 or 1 here)
            logger.debug(f"Cache DELETE for key: {key}. Result: {result}")
            return True # Return True even if key didn't exist (idempotent)
        except aioredis.RedisError as e:
            logger.error(f"Redis DELETE error for key {key}: {e}")
            return False
        except Exception as e:
             logger.error(f"Unexpected error during cache DELETE for key {key}: {e}")
             return False

    async def clear_all(self) -> bool:
        """
        Flushes the entire Redis database associated with the client.
        USE WITH EXTREME CAUTION!
        """
        if not self.redis:
            logger.warning("Attempted cache CLEAR ALL when Redis client is unavailable.")
            return False
        try:
            logger.warning("Executing FLUSHDB on Redis cache!")
            await self.redis.flushdb()
            logger.info("Redis cache flushed successfully.")
            return True
        except aioredis.RedisError as e:
            logger.error(f"Redis FLUSHDB error: {e}")
            return False
        except Exception as e:
             logger.error(f"Unexpected error during cache FLUSHDB: {e}")
             return False

# --- Optional: No-operation Cache Service ---
# class NoOpCacheService:
#     """A cache service that does nothing, useful when Redis is unavailable."""
#     def __init__(self):
#         logger.warning("Using NoOpCacheService. Caching is disabled.")

#     async def get(self, key: str) -> Optional[Any]:
#         logger.debug(f"NoOpCache GET ignored for key: {key}")
#         return None

#     async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
#         logger.debug(f"NoOpCache SET ignored for key: {key}")
#         return True # Indicate success to avoid breaking logic relying on set return

#     async def delete(self, key: str) -> bool:
#         logger.debug(f"NoOpCache DELETE ignored for key: {key}")
#         return True

#     async def clear_all(self) -> bool:
#          logger.debug("NoOpCache CLEAR ALL ignored.")
#          return True