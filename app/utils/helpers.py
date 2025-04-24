import asyncio
from typing import List, Any, Callable, TypeVar, Coroutine
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from loguru import logger
from pydantic import BaseModel

# --- Retry Decorator ---

# Define specific exceptions you might want to retry on (e.g., network issues)
# from httpx import NetworkError, TimeoutException
# RETRYABLE_EXCEPTIONS = (NetworkError, TimeoutException, asyncio.TimeoutError)

T = TypeVar('T')

def async_retry_request(
    attempts: int = 3,
    initial_wait: int = 1, # seconds
    max_wait: int = 10, # seconds
    # exceptions_to_retry: tuple = RETRYABLE_EXCEPTIONS
    exceptions_to_retry: tuple = (Exception,) # Retry on any Exception by default - be cautious
) -> Callable[[Callable[..., Coroutine[Any, Any, T]]], Callable[..., Coroutine[Any, Any, T]]]:
    """
    Decorator for retrying async functions with exponential backoff.

    Args:
        attempts: Maximum number of attempts.
        initial_wait: Minimum wait time between retries.
        max_wait: Maximum wait time between retries.
        exceptions_to_retry: Tuple of exception types to trigger a retry.
    """
    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        @retry(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=1, min=initial_wait, max=max_wait),
            retry=retry_if_exception_type(exceptions_to_retry),
            before_sleep=lambda retry_state: logger.warning(
                f"Retrying {func.__name__} due to {retry_state.outcome.exception()}. "
                f"Attempt {retry_state.attempt_number}/{attempts}. "
                f"Waiting {retry_state.next_action.sleep:.2f}s..."
            )
        )
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            logger.debug(f"Executing function '{func.__name__}' with retry logic...")
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# --- Pagination ---

class PageParams(BaseModel):
    skip: int = Field(0, ge=0, description="Number of items to skip")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of items to return") # Add sensible max limit

def paginate(items: List[Any], params: PageParams) -> List[Any]:
    """
    Paginates a list of items based on skip and limit parameters.

    Args:
        items: The list of items to paginate.
        params: A PageParams object containing skip and limit.

    Returns:
        The sliced list representing the requested page.
    """
    start = params.skip
    # Ensure end does not exceed list bounds gracefully
    end = min(start + params.limit, len(items))
    # Handle cases where skip might be larger than the list size
    if start >= len(items):
        return []
    logger.debug(f"Paginating list: total={len(items)}, skip={params.skip}, limit={params.limit}. Returning slice [{start}:{end}]")
    return items[start:end]