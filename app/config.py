import os
from typing import List, Optional
from pydantic import BaseSettings, Field, AnyHttpUrl, RedisDsn, validator
import logging

# Define log levels explicitly for validation
LOG_LEVELS = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"]

class Settings(BaseSettings):
    # App Core
    APP_NAME: str = "LangChain FastAPI Service"
    APP_VERSION: str = "2.1.0"
    LOG_LEVEL: str = "INFO"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = Field(..., env="SECRET_KEY") # For JWT, CSRF etc. - MUST be set

    # CORS
    CORS_ORIGINS: List[AnyHttpUrl] = ["http://localhost:3000", "http://localhost:8080"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]

    # Rate Limiting (example using slowapi)
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_DEFAULT: str = "100/minute"

    # Authentication
    API_KEY: str = Field(..., env="API_KEY") # Simple API Key for internal services/testing
    # For robust auth, add JWT settings:
    # JWT_ALGORITHM: str = "HS256"
    # JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # OpenAI
    OPENAI_API_KEY: Optional[str] = Field(None, env="OPENAI_API_KEY")
    OPENAI_LLM_MODEL: str = "gpt-3.5-turbo"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-ada-002"

    # Gemini
    GEMINI_API_KEY: Optional[str] = Field(None, env="GEMINI_API_KEY")
    GEMINI_LLM_MODEL: str = "gemini-pro" # Adjust model name as needed
    # GEMINI_USE_VERTEXAI: bool = False # Optional: If using Vertex AI

    # Groq
    GROQ_API_KEY: Optional[str] = Field(None, env="GROQ_API_KEY")
    GROQ_LLM_MODEL: str = "llama3-8b-8192"

    # Default Provider (can be 'openai', 'gemini', 'groq')
    DEFAULT_LLM_PROVIDER: str = "openai"

    # Vector Store
    VECTORSTORE_TYPE: str = "chroma" # e.g., 'chroma', 'faiss', 'pinecone'
    VECTORSTORE_PATH: str = "./chroma_db"
    # Add other vector store specific configs (e.g., Pinecone API key/env)

    # Redis Cache
    REDIS_URL: RedisDsn = "redis://localhost:6379/0"
    CACHE_EXPIRATION_SECONDS: int = 300

    # S3 Storage
    S3_ENABLED: bool = False
    S3_ENDPOINT_URL: Optional[AnyHttpUrl] = None
    S3_ACCESS_KEY: Optional[str] = None
    S3_SECRET_KEY: Optional[str] = None
    S3_BUCKET_NAME: Optional[str] = None
    S3_REGION: Optional[str] = "us-east-1" # Or your relevant region

    # Telemetry (OTLP endpoint for Jaeger, Tempo, etc.)
    OTEL_EXPORTER_OTLP_ENDPOINT: Optional[AnyHttpUrl] = None
    OTEL_SERVICE_NAME: str = "fastapi-langchain-service"

    @validator('LOG_LEVEL', pre=True, always=True)
    def validate_log_level(cls, value):
        upper_value = value.upper()
        if upper_value not in LOG_LEVELS:
            raise ValueError(f"Invalid LOG_LEVEL: {value}. Must be one of {LOG_LEVELS}")
        return upper_value

    @validator('DEFAULT_LLM_PROVIDER', pre=True, always=True)
    def validate_default_provider(cls, value, values):
        provider = value.lower()
        if provider == 'openai' and not values.get('OPENAI_API_KEY'):
            logging.warning("Default provider is OpenAI, but OPENAI_API_KEY is not set.")
        elif provider == 'gemini' and not values.get('GEMINI_API_KEY'):
             logging.warning("Default provider is Gemini, but GEMINI_API_KEY is not set.")
        elif provider == 'groq' and not values.get('GROQ_API_KEY'):
             logging.warning("Default provider is Groq, but GROQ_API_KEY is not set.")
        # Add more provider checks if needed
        return provider

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Add prefix if desired, e.g., env_prefix = "APP_"

# Global settings instance
settings = Settings()