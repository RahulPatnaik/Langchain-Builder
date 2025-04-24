import time
import uuid
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from loguru import logger

from app.config import settings
from app.utils.logging_config import setup_logging
from app.utils.telemetry import init_tracer, instrument_app
from app.utils.error_handlers import register_exception_handlers
from app.dependencies import create_redis_pool, create_s3_client # Startup dependencies
from app.security.rate_limit import limiter # Import configured limiter
from app.routers import chat, documents, utils # Import API routers

# Initialize Logging FIRST
setup_logging()

# Initialize Tracer
init_tracer()

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Enhanced modular FastAPI service for LangChain applications.",
    openapi_url=f"{settings.API_V1_STR}/openapi.json" # Versioned OpenAPI schema
)

# Rate Limiting State
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- Middleware Stack ---
# ORDER MATTERS HERE

# 1. Tracing/Context Middleware (Inject Request ID early)
@app.middleware("http")
async def add_context_vars(request: Request, call_next):
    start_time = time.perf_counter()
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    # Set context vars for logging/tracing if needed (Loguru can often capture this automatically)
    # with logger.contextualize(request_id=request_id): # Example if needed explicitly
    #     response = await call_next(request)

    response = await call_next(request) # Process request

    process_time = time.perf_counter() - start_time
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{process_time:.4f}" # Add process time header

    # Log request details after processing
    logger.info(
        f"Request: {request.method} {request.url.path} "
        f"Status: {response.status_code} "
        f"Client: {request.client.host} Agent: {request.headers.get('user-agent', 'N/A')} "
        f"Time: {process_time:.4f}s ID: {request_id}"
    )
    return response


# 2. CORS Middleware
if settings.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.CORS_ORIGINS], # Pydantic v1 needs str conversion
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
    )
    logger.info(f"CORS enabled for origins: {settings.CORS_ORIGINS}")

# --- Event Handlers (Startup/Shutdown) ---
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup...")
    await create_redis_pool() # Initialize Redis pool
    await create_s3_client()  # Initialize S3 client
    # Initialize other resources like database connections, ML models etc.
    logger.info("Application startup complete.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown...")
    # Clean up resources
    redis_pool = await create_redis_pool() # Get pool instance
    if redis_pool:
        await redis_pool.close()
        # await redis_pool.wait_closed() # Optional: wait for connections to close
        logger.info("Redis connection pool closed.")

    s3_client = await create_s3_client() # Get S3 client instance
    if s3_client:
        await s3_client.__aexit__(None, None, None) # Properly close async client
        logger.info("S3 client closed.")
    logger.info("Application shutdown complete.")


# --- Prometheus Metrics ---
# Exposes /metrics endpoint automatically
instrumentator = Instrumentator(
    should_group_status_codes=False, # Group 2xx, 3xx etc. ?
    should_instrument_requests_inprogress=True,
    excluded_handlers=[f"{settings.API_V1_STR}/health", "/metrics"], # Don't track metrics/health itself
    inprogress_labels=True,
).instrument(app)

# Instrument with OpenTelemetry *after* Prometheus setup potentially
instrument_app(app)

# --- API Routers ---
# Include routers from the 'routers' module
app.include_router(utils.router) # Health, metrics etc. (no prefix or specific prefix)
app.include_router(chat.router, prefix=settings.API_V1_STR, tags=["Chat"])
app.include_router(documents.router, prefix=settings.API_V1_STR, tags=["Documents & RAG"])
# Add other routers...

# --- Custom Error Handlers ---
# Register custom handlers AFTER adding routers if they override default behavior
register_exception_handlers(app)

# --- Root Endpoint ---
@app.get("/", tags=["Root"])
async def read_root():
    return {
        "message": f"Welcome to {settings.APP_NAME} v{settings.APP_VERSION}",
        "docs": "/docs",
        "redoc": "/redoc"
    }

# Note: run.py would handle the uvicorn execution
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, log_level=settings.LOG_LEVEL.lower())