from functools import lru_cache
from typing import Optional
from loguru import logger

# Langchain Imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI #, GoogleGenerativeAIEmbeddings (If needed)
from langchain_groq import ChatGroq

# --- Individual Provider Functions ---

@lru_cache() # Cache LLM instances based on parameters
def get_openai_chat_llm(api_key: str, model_name: str, temperature: float = 0.7, **kwargs) -> ChatOpenAI:
    logger.debug(f"Creating OpenAI Chat LLM: model={model_name}, temp={temperature}")
    return ChatOpenAI(
        openai_api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        **kwargs # Pass other params like max_tokens etc.
    )

@lru_cache()
def get_gemini_chat_llm(api_key: str, model_name: str, temperature: float = 0.7, **kwargs) -> ChatGoogleGenerativeAI:
    logger.debug(f"Creating Gemini Chat LLM: model={model_name}, temp={temperature}")
    # Check for specific Gemini parameters if needed
    return ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model=model_name, # Gemini uses 'model'
        temperature=temperature,
        convert_system_message_to_human=True, # Often needed for Gemini compatibility
        **kwargs
    )

@lru_cache()
def get_groq_chat_llm(api_key: str, model_name: str, temperature: float = 0.7, **kwargs) -> ChatGroq:
    logger.debug(f"Creating Groq Chat LLM: model={model_name}, temp={temperature}")
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        **kwargs
    )

# --- Embedding Provider Functions ---

@lru_cache()
def get_openai_embeddings(api_key: str, model_name: str, **kwargs) -> OpenAIEmbeddings:
    logger.debug(f"Creating OpenAI Embeddings: model={model_name}")
    return OpenAIEmbeddings(
        openai_api_key=api_key,
        model=model_name,
        **kwargs
    )

# Add Gemini embeddings if needed and available in langchain_google_genai
# @lru_cache()
# def get_gemini_embeddings(api_key: str, model_name: str, **kwargs) -> GoogleGenerativeAIEmbeddings:
#     logger.debug(f"Creating Gemini Embeddings: model={model_name}")
#     return GoogleGenerativeAIEmbeddings(google_api_key=api_key, model=model_name, **kwargs)


# --- Optional: Factory Pattern (Alternative to direct functions in dependencies.py) ---
# from app.config import Settings

# class LLMProviderFactory:
#     def __init__(self, settings: Settings):
#         self.settings = settings

#     async def get_llm(self, provider_name: Optional[str] = None) -> BaseChatModel:
#         provider = (provider_name or self.settings.DEFAULT_LLM_PROVIDER).lower()
#         logger.debug(f"Factory getting LLM for provider: {provider}")

#         if provider == "openai":
#             if not self.settings.OPENAI_API_KEY:
#                 raise ValueError("OpenAI API key not configured")
#             # Consider async creation if needed, but instance creation is usually fast
#             return get_openai_chat_llm(self.settings.OPENAI_API_KEY, self.settings.OPENAI_LLM_MODEL)
#         elif provider == "gemini":
#             if not self.settings.GEMINI_API_KEY:
#                 raise ValueError("Gemini API key not configured")
#             return get_gemini_chat_llm(self.settings.GEMINI_API_KEY, self.settings.GEMINI_LLM_MODEL)
#         elif provider == "groq":
#             if not self.settings.GROQ_API_KEY:
#                  raise ValueError("Groq API key not configured")
#             return get_groq_chat_llm(self.settings.GROQ_API_KEY, self.settings.GROQ_LLM_MODEL)
#         else:
#             raise ValueError(f"Unsupported LLM provider: {provider}")

#     async def get_embeddings(self, provider_name: Optional[str] = None) -> Embeddings:
#         # Similar logic for embeddings based on config
#         # For simplicity, assuming OpenAI embeddings are default/primary for now
#         logger.debug(f"Factory getting embeddings (defaulting to OpenAI)")
#         if self.settings.OPENAI_API_KEY:
#             return get_openai_embeddings(self.settings.OPENAI_API_KEY, self.settings.OPENAI_EMBEDDING_MODEL)
#         # Add logic for other embedding providers if needed
#         else:
#             raise ValueError("OpenAI API key not configured for embeddings")