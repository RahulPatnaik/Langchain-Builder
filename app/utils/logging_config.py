import logging
import sys
import json
from loguru import logger # Using Loguru for better structured logging
from app.config import settings

class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

def setup_logging():
    # Remove default handlers
    logger.remove()

    # Add console handler with structured JSON output if desired, or keep human-readable
    # For JSON: logger.add(sys.stderr, serialize=True, level=settings.LOG_LEVEL)
    logger.add(
        sys.stderr,
        level=settings.LOG_LEVEL,
        format=( # Example human-readable format
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True # Auto-detects if the terminal supports color
    )

    # Intercept standard logging messages toward Loguru sinks
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True) # Capture all stdlib logs
    logging.getLogger("uvicorn.access").handlers = [InterceptHandler()] # Route uvicorn access logs
    logging.getLogger("uvicorn.error").handlers = [InterceptHandler()] # Route uvicorn errors

    # Configure other library log levels if needed
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("aioredis").setLevel(logging.WARNING)

    logger.info(f"Logging configured with level: {settings.LOG_LEVEL}")