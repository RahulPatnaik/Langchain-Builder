from functools import lru_cache
from typing import Optional, AsyncGenerator

import aioredis
from fastapi import Depends, HTTPException, status
from loguru import logger

# Langchain specific imports (adjust based on actual usage)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

# Example for other providers
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_groq import ChatGroq

# S3
import aiobotocore.session # Async S3
from types import SimpleNamespace # For mocking S3 client if needed

from app.config import settings
from app.services.llm_provider import (
    get_openai_chat_llm, get_gemini_chat_llm, get_groq_chat_llm,
    get_openai_embeddings, # Add others: get_gemini_embeddings etc.
    LLMProviderFactory # Optional factory pattern
)
from app.services.vector_store import get_chroma_vectorstore # Add others: get_faiss_vectorstore
from app.services.cache import RedisCacheService

# --- LLM Providers ---

# Use a factory or specific functions based on preference
# @lru_cache() # Cache the factory instance
# def get_llm_factory() -> LLMProviderFactory:
#     return LLMProviderFactory(settings)

# async def get_llm(factory: LLMProviderFactory = Depends(get_llm_factory)) -> BaseChatModel:
#     try:
#         return await factory.get_llm(settings.DEFAULT_LLM_PROVIDER)
#     except ValueError as e:
#         logger.error(f"Failed to get LLM provider '{settings.DEFAULT_LLM_PROVIDER}': {e}")
#         raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="LLM provider not configured or unavailable")

# --- OR directly use specific functions ---
async def get_chat_llm() -> BaseChatModel:
    provider = settings.DEFAULT_LLM_PROVIDER.lower()
    logger.debug(f"Getting chat LLM for provider: {provider}")
    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "OpenAI API key not configured")
        return get_openai_chat_llm(settings.OPENAI_API_KEY, settings.OPENAI_LLM_MODEL)
    elif provider == "gemini":
        if not settings.GEMINI_API_KEY:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Gemini API key not configured")
        return get_gemini_chat_llm(settings.GEMINI_API_KEY, settings.GEMINI_LLM_MODEL)
    elif provider == "groq":
         if not settings.GROQ_API_KEY:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Groq API key not configured")
         return get_groq_chat_llm(settings.GROQ_API_KEY, settings.GROQ_LLM_MODEL)
    else:
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, f"LLM provider '{provider}' not supported")


# --- Embeddings Provider ---
@lru_cache()
def get_embeddings() -> Embeddings:
    # Simple example, expand similarly to get_chat_llm for multiple providers
    if settings.OPENAI_API_KEY:
         logger.debug(f"Using OpenAI embeddings model: {settings.OPENAI_EMBEDDING_MODEL}")
         return get_openai_embeddings(settings.OPENAI_API_KEY, settings.OPENAI_EMBEDDING_MODEL)
    # Add other providers (Gemini, local, etc.) based on config
    else:
         logger.error("No embedding provider API key found (checked OpenAI).")
         raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Embedding model provider not configured")

# --- Vector Store ---
@lru_cache() # Cache the vector store instance
def get_vector_store(embeddings: Embeddings = Depends(get_embeddings)) -> Chroma: # Return specific type or a BaseRetriever/BaseVectorStore
    store_type = settings.VECTORSTORE_TYPE.lower()
    logger.debug(f"Getting vector store of type: {store_type}")
    if store_type == "chroma":
        return get_chroma_vectorstore(
            persist_directory=settings.VECTORSTORE_PATH,
            embedding_function=embeddings
        )
    # Add other vector stores (FAISS, Pinecone, etc.)
    # elif store_type == "faiss":
    #     return get_faiss_vectorstore(...)
    else:
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, f"Vector store type '{store_type}' not supported")

# --- Redis Cache ---
# Global Redis pool instance (managed connection pool)
_redis_pool: Optional[aioredis.Redis] = None

async def create_redis_pool():
    global _redis_pool
    if _redis_pool is None:
        logger.info(f"Creating Redis connection pool for URL: {settings.REDIS_URL}")
        try:
            # Use connection pool for better performance
            _redis_pool = await aioredis.from_url(
                str(settings.REDIS_URL),
                encoding="utf-8",
                decode_responses=True,
                health_check_interval=30 # Check connection health periodically
            )
            # Test connection
            await _redis_pool.ping()
            logger.info("Successfully connected to Redis.")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            _redis_pool = None # Ensure it's None if connection fails
            # Depending on criticality, you might raise an exception here to prevent app startup
            # raise RuntimeError("Failed to establish Redis connection pool") from e
    return _redis_pool

async def get_redis() -> Optional[aioredis.Redis]:
    """Provides a Redis connection from the pool. Returns None if pool creation failed."""
    if _redis_pool is None:
        # Attempt to create pool again if it failed during startup (optional)
        # await create_redis_pool()
        # if _redis_pool is None:
        logger.warning("Redis pool is not available.")
        return None
    return _redis_pool

async def get_cache_service(redis: Optional[aioredis.Redis] = Depends(get_redis)) -> RedisCacheService:
    if redis is None:
        # Provide a dummy/no-op cache service if Redis is unavailable
        logger.warning("Redis not available, using NoOp cache service.")
        # return NoOpCacheService() # Implement a NoOp service if needed
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Cache service unavailable")
    return RedisCacheService(redis, default_ttl=settings.CACHE_EXPIRATION_SECONDS)


# --- S3 Client ---
# Global async S3 client instance
_s3_client = None

async def create_s3_client():
    global _s3_client
    if settings.S3_ENABLED and _s3_client is None:
        logger.info(f"Creating S3 client for endpoint: {settings.S3_ENDPOINT_URL}")
        try:
            session = aiobotocore.session.get_session()
            # Use SimpleNamespace for mocking in tests if needed
            # _s3_client = SimpleNamespace(upload_file=async_mock_fn, download_file=async_mock_fn, ...)
            _s3_client = await session.create_client(
                's3',
                endpoint_url=str(settings.S3_ENDPOINT_URL) if settings.S3_ENDPOINT_URL else None,
                aws_access_key_id=settings.S3_ACCESS_KEY,
                aws_secret_access_key=settings.S3_SECRET_KEY,
                region_name=settings.S3_REGION
            ).__aenter__() # Enter the async context manager
            logger.info("S3 client created successfully.")
            # Optionally test connection: await _s3_client.list_buckets()
        except Exception as e:
            logger.error(f"Failed to create S3 client: {e}")
            _s3_client = None
            # raise RuntimeError("Failed to establish S3 connection") from e
    elif not settings.S3_ENABLED:
        logger.info("S3 is disabled in settings.")
    return _s3_client

async def get_s3_client():
    """Provides an async S3 client. Returns None if disabled or creation failed."""
    if _s3_client is None and settings.S3_ENABLED:
        # await create_s3_client() # Potentially retry creation
        # if _s3_client is None:
        logger.warning("S3 client is not available.")
        return None
    return _s3_client

# --- Langchain Chains (Example) ---
# You might create chains directly in endpoints or abstract them here/in services
# async def get_qa_chain(
#     llm: BaseChatModel = Depends(get_chat_llm),
#     vector_store: Chroma = Depends(get_vector_store) # Use specific type or BaseRetriever
# ) -> RetrievalQA:
#      # Note: Ensure the chain components support async if running fully async
#     logger.debug("Creating RetrievalQA chain")
#     retriever = vector_store.as_retriever()
#     # Check if async is supported by the specific chain type
#     # Might need to use methods like `ainvoke` later
#     return RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff", # Or other types like "map_reduce"
#         retriever=retriever,
#         return_source_documents=True # Optional: return sources
#     )

# Add other dependency providers as needed (e.g., databases, external APIs)