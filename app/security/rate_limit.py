from slowapi import Limiter
from slowapi.util import get_remote_address
from loguru import logger

from app.config import settings

# Create a limiter instance
# It uses the client's remote address (IP) as the key by default
limiter = Limiter(
    key_func=get_remote_address,
    enabled=settings.RATE_LIMIT_ENABLED,
    # Storage can be configured (e.g., "redis://localhost:6379/1" for redis)
    # storage_uri=str(settings.REDIS_URL) # Example if using Redis for limits
    # By default, it uses in-memory storage, which is not suitable for multi-process/multi-instance deployments.
    # strategy="fixed-window" # or "moving-window"
    default_limits=[settings.RATE_LIMIT_DEFAULT] # e.g., "100/minute", "20/second"
)

if settings.RATE_LIMIT_ENABLED:
     logger.info(f"Rate limiting enabled with default limit: {settings.RATE_LIMIT_DEFAULT}")
else:
     logger.info("Rate limiting is disabled.")

# You can apply rate limits per-route using:
# @router.get("/limited")
# @limiter.limit("5/minute") # Specific limit for this route
# async def limited_route(request: Request):
#     return {"message": "This route has a specific limit"}