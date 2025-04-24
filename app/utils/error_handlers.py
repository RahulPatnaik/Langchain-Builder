from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError, HTTPException
from pydantic import ValidationError # Import Pydantic's own error
from loguru import logger

# Consistent error response format
class ErrorResponse(BaseModel):
    detail: Any

def register_exception_handlers(app: FastAPI):
    """Registers custom exception handlers for the FastAPI app."""

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handles FastAPI's built-in HTTPExceptions."""
        logger.warning(
            f"HTTPException: Status={exc.status_code} Detail='{exc.detail}' Path={request.url.path}"
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handles Pydantic validation errors for request bodies, query params, etc."""
        # Log the detailed validation errors
        logger.warning(
            f"RequestValidationError: Path={request.url.path} Errors='{exc.errors()}'"
        )
        # Provide a user-friendly error message
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": "Validation Error", "errors": exc.errors()},
        )

    # Optional: Handler for Pydantic ValidationErrors outside requests (e.g., in settings)
    @app.exception_handler(ValidationError)
    async def pydantic_validation_handler(request: Request, exc: ValidationError):
         """Handles Pydantic ValidationErrors occurring outside of request validation."""
         logger.error(
            f"Pydantic ValidationError (outside request): Path={request.url.path} Errors='{exc.errors()}'"
         )
         return JSONResponse(
             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, # Or 400 if appropriate context
             content={"detail": "Internal configuration or data validation error.", "errors": exc.errors()},
         )


    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """Handles any other unhandled exceptions."""
        # Log the full exception traceback
        logger.error(
            f"Unhandled Exception: Type={type(exc).__name__} Message='{exc}' Path={request.url.path}",
            exc_info=True # Includes traceback
        )
        # Return a generic 500 error response
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal Server Error"},
        )

    logger.info("Custom exception handlers registered.")