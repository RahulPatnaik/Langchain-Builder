from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
import aioredis
from langchain_community.vectorstores import Chroma
import httpx # For external checks

from app.config import settings, Settings
from app.dependencies import get_redis, get_vector_store, get_s3_client # Import dependency getters
from app.security.auth import get_api_key # Optional: protect utils endpoints


router = APIRouter(tags=["Utilities"])

# --- Health Check Component Functions ---

async def check_redis_health(redis: aioredis.Redis = Depends(get_redis)) -> bool:
    if not redis: return False
    try:
        return await redis.ping()
    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")
        return False

def check_vector_store_health(vector_store: Chroma = Depends(get_vector_store)) -> bool:
    # Basic check: try getting collection count or a small query
    # This is synchronous for Chroma usually
    if not vector_store: return False
    try:
        # Example: Check if we can get metadata for a known collection or just count
        vector_store.count() # Throws exception if underlying connection fails
        # or vector_store.get() or vector_store.peek()
        return True
    except Exception as e:
        logger.warning(f"Vector store health check failed: {e}")
        return False

async def check_s3_health(s3_client = Depends(get_s3_client)) -> bool:
    if not settings.S3_ENABLED: return True # Healthy if disabled
    if not s3_client: return False
    try:
        # Perform a simple operation, like listing buckets (requires permissions)
        # or head_bucket if bucket name is known
        if settings.S3_BUCKET_NAME:
            await s3_client.head_bucket(Bucket=settings.S3_BUCKET_NAME)
            return True
        else:
            # Fallback if no bucket name configured, less reliable
            await s3_client.list_buckets()
            return True
    except Exception as e:
        logger.warning(f"S3 health check failed: {e}")
        return False

# Add checks for LLM providers if desired (e.g., pinging OpenAI status endpoint)
# async def check_openai_health() -> bool:
#     try:
#         async with httpx.AsyncClient() as client:
#             # Note: OpenAI doesn't have a public simple /health endpoint.
#             # Could try listing models if API key is available, but might incur cost/rate limit.
#             # A simple check might be just ensuring the API key is set.
#             return bool(settings.OPENAI_API_KEY)
#     except Exception:
#         return False


# --- Health Endpoint ---

@router.get("/health", summary="Perform health checks on the service and dependencies")
async def health_check(
    # Inject dependencies needed for checks
    redis_healthy: bool = Depends(check_redis_health),
    vector_store_healthy: bool = Depends(check_vector_store_health),
    s3_healthy: bool = Depends(check_s3_health)
    # openai_healthy: bool = Depends(check_openai_health),
):
    """
    Checks the status of the main application and its core dependencies
    (Cache, Vector Store, Object Storage).
    """
    status_code = 200
    dependencies = {
        "cache_redis": "ok" if redis_healthy else "error",
        "vector_store_chroma": "ok" if vector_store_healthy else "error",
        "storage_s3": "ok" if s3_healthy else ("disabled" if not settings.S3_ENABLED else "error"),
        # "llm_openai": "ok" if openai_healthy else "error",
    }
    overall_status = "ok"

    for key, value in dependencies.items():
        if value == "error":
            overall_status = "error"
            status_code = 503 # Service Unavailable

    response = {
        "status": overall_status,
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "dependencies": dependencies
    }

    if overall_status == "error":
        logger.warning(f"Health check failed: {response}")
        raise HTTPException(status_code=status_code, detail=response)
    else:
        logger.debug(f"Health check successful: {response}")
        return response

# --- Optional: Configuration Endpoint ---
@router.get(
    "/config",
    summary="Show non-sensitive configuration (requires auth)",
    dependencies=[Depends(get_api_key)] # Secure this endpoint
)
async def get_configuration(app_settings: Settings = Depends(lambda: settings)):
    """
    Returns a subset of the application configuration, excluding secrets.
    """
    # Be VERY careful what you expose here. Exclude all secrets.
    safe_config = {
        "app_name": app_settings.APP_NAME,
        "app_version": app_settings.APP_VERSION,
        "log_level": app_settings.LOG_LEVEL,
        "default_llm_provider": app_settings.DEFAULT_LLM_PROVIDER,
        "openai_llm_model": app_settings.OPENAI_LLM_MODEL if app_settings.OPENAI_API_KEY else None,
        "gemini_llm_model": app_settings.GEMINI_LLM_MODEL if app_settings.GEMINI_API_KEY else None,
        "groq_llm_model": app_settings.GROQ_LLM_MODEL if app_settings.GROQ_API_KEY else None,
        "vectorstore_type": app_settings.VECTORSTORE_TYPE,
        "redis_url_host": app_settings.REDIS_URL.host if app_settings.REDIS_URL else None,
        "s3_enabled": app_settings.S3_ENABLED,
        "s3_endpoint_url": str(app_settings.S3_ENDPOINT_URL) if app_settings.S3_ENDPOINT_URL else None,
        "s3_bucket_name": app_settings.S3_BUCKET_NAME,
        "otel_exporter_otlp_endpoint": str(app_settings.OTEL_EXPORTER_OTLP_ENDPOINT) if app_settings.OTEL_EXPORTER_OTLP_ENDPOINT else None,
        "otel_service_name": app_settings.OTEL_SERVICE_NAME,
        "rate_limit_enabled": app_settings.RATE_LIMIT_ENABLED,
        "rate_limit_default": app_settings.RATE_LIMIT_DEFAULT,
        "cors_origins": [str(o) for o in app_settings.CORS_ORIGINS],
    }
    return safe_config