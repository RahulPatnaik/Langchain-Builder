import uvicorn
from loguru import logger

from app.main import app # Import the FastAPI app instance
from app.config import settings

if __name__ == "__main__":
    # Log level for uvicorn should ideally match the app's log level
    log_level = settings.LOG_LEVEL.lower()
    # Ensure log level is valid for uvicorn
    valid_uvicorn_levels = ["critical", "error", "warning", "info", "debug", "trace"]
    if log_level not in valid_uvicorn_levels:
        logger.warning(f"Invalid LOG_LEVEL '{settings.LOG_LEVEL}' for Uvicorn, defaulting to 'info'.")
        log_level = "info"

    logger.info(f"Starting Uvicorn server on host=0.0.0.0, port=8000, log_level={log_level}")
    # Use reload=True for development, False for production
    # Use workers=N for production (where N is typically 2 * num_cores + 1)
    uvicorn.run(
        "app.main:app", # Point to the app instance location
        host="0.0.0.0",
        port=8000,
        log_level=log_level,
        reload=True # Set reload=False for production
        # workers=4 # Enable for production
    )