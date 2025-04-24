from functools import lru_cache
from loguru import logger
import os

# Langchain Imports
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
# from langchain_community.vectorstores import FAISS # Example for another type
# from langchain_pinecone import Pinecone # Example for Pinecone

from app.config import settings # Import to access path settings

@lru_cache() # Cache based on path and embedding instance (or its hash)
def get_chroma_vectorstore(
    persist_directory: str,
    embedding_function: Embeddings,
    collection_name: str = "default" # Default collection name if not specified elsewhere
) -> Chroma:
    """
    Initializes and returns a Chroma vector store instance.
    Handles persistence directory creation.
    """
    logger.info(f"Initializing Chroma vector store at: {persist_directory} with collection: {collection_name}")

    # Ensure the persist directory exists
    if not os.path.exists(persist_directory):
        logger.warning(f"Chroma persist directory '{persist_directory}' not found. Creating.")
        try:
            os.makedirs(persist_directory)
        except OSError as e:
            logger.error(f"Failed to create Chroma directory {persist_directory}: {e}")
            raise  # Re-raise the exception as store creation will likely fail

    try:
        # Initialize Chroma. It handles loading existing data if the directory has it.
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
            collection_name=collection_name
            # Add other Chroma specific args if needed e.g. client_settings
        )
        # Optional: Log count on initialization
        count = vector_store._collection.count()
        logger.info(f"Chroma vector store initialized/loaded successfully. Collection '{collection_name}' contains {count} vectors.")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to initialize Chroma vector store: {e}", exc_info=True)
        raise # Re-raise to indicate failure

# --- Example for FAISS (if you were to add it) ---
# @lru_cache()
# def get_faiss_vectorstore(
#     index_path: Optional[str], # FAISS often saved as a file
#     embedding_function: Embeddings,
#     # Note: FAISS loading/saving requires more manual handling usually
# ) -> FAISS:
#     logger.info(f"Attempting to load FAISS index from: {index_path}")
#     if index_path and os.path.exists(index_path):
#         try:
#             # FAISS loading depends on how it was saved (local, S3, etc.)
#             # This is a basic example for local loading
#             vector_store = FAISS.load_local(
#                 folder_path=os.path.dirname(index_path), # Assuming index_path is 'folder/index.faiss'
#                 embeddings=embedding_function,
#                 index_name=os.path.basename(index_path).replace(".faiss", "") # e.g., 'index'
#             )
#             logger.info("FAISS index loaded successfully.")
#             return vector_store
#         except Exception as e:
#             logger.error(f"Failed to load FAISS index from {index_path}: {e}", exc_info=True)
#             # Decide whether to fallback to creating a new one or raise
#             raise
#     else:
#         logger.warning(f"FAISS index not found at {index_path}. A new in-memory store will be used (if operations proceed).")
#         # Return an empty store? This depends on desired behavior.
#         # FAISS needs documents to be created initially, unlike Chroma's persistent dir.
#         # This function might need adjustment based on workflow (e.g., raise error if load fails)
#         raise FileNotFoundError(f"FAISS index file not found: {index_path}")

# --- Add other vector store initializers as needed (Pinecone, Weaviate, etc.) ---