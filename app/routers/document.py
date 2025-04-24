import os
import shutil
import tempfile
from typing import List, Optional
from uuid import uuid4

from fastapi import (APIRouter, Depends, HTTPException, UploadFile, File, Form,
                     BackgroundTasks, status)
from loguru import logger
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import ( UnstructuredPDFLoader,
                                                   TextLoader,
                                                   UnstructuredFileLoader )


from app.models.schemas import ( IndexRequest, IndexResponse,
                                 QueryRequest, QueryResponse, DocumentMetadata )
from app.dependencies import get_vector_store, get_embeddings, get_chat_llm, get_s3_client
from app.config import settings
from app.services.storage import upload_to_s3, download_from_s3
from app.security.auth import get_api_key # Or your preferred auth dependency

router = APIRouter()

# --- Helper Function for Document Processing ---

async def process_and_index_document(
    file_path: str,
    filename: str,
    vector_store: Chroma,
    embeddings: Embeddings,
    collection_name: str = "default", # Example: Use collection names
    metadata: Optional[dict] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 100
):
    """Loads, splits, embeds, and indexes a document asynchronously."""
    logger.info(f"Processing document: {filename} for collection '{collection_name}'")
    base_metadata = {"source": filename}
    if metadata:
        base_metadata.update(metadata)

    try:
        # Determine loader based on file extension
        _, ext = os.path.splitext(filename.lower())
        if ext == ".pdf":
            loader = UnstructuredPDFLoader(file_path, mode="single") # Or "elements"
        elif ext in [".txt", ".md", ".py", ".json", ".yaml", ".html", ".csv"]:
            # Use TextLoader for common text types, might need encoding handling
            try:
                loader = TextLoader(file_path, encoding="utf-8")
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 decoding failed for {filename}, trying latin-1")
                loader = TextLoader(file_path, encoding="latin-1") # Fallback encoding
        else:
            # Fallback for other types supported by unstructured
            logger.warning(f"Using generic UnstructuredFileLoader for extension '{ext}'")
            loader = UnstructuredFileLoader(file_path, mode="single")

        # Load documents (sync operation, potentially long)
        # Consider running this in a thread pool for large files in a highly async app
        # from fastapi.concurrency import run_in_threadpool
        # documents = await run_in_threadpool(loader.load)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} document sections from {filename}.")

        if not documents:
            logger.warning(f"No content loaded from document: {filename}")
            return

        # Update metadata for all loaded documents
        for doc in documents:
            doc.metadata.update(base_metadata)
            # Ensure metadata values are suitable for Chroma (str, int, float, bool)
            for key, value in doc.metadata.items():
                 if not isinstance(value, (str, int, float, bool)):
                     doc.metadata[key] = str(value)


        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"Split document into {len(split_docs)} chunks.")

        if not split_docs:
            logger.warning(f"Document splitting resulted in 0 chunks: {filename}")
            return

        # Add to vector store (sync operation)
        # Again, consider thread pool for true async
        # await run_in_threadpool(
        #     vector_store.add_documents,
        #     documents=split_docs,
        #     embedding=embeddings, # Pass embedding function if needed by add_documents
        #     collection_name=collection_name # If Chroma client supports it directly
        # )
        # Standard Chroma add_documents:
        ids = [str(uuid4()) for _ in split_docs] # Generate unique IDs for each chunk
        vector_store.add_documents(documents=split_docs, embedding=embeddings, ids=ids)

        # Persist changes if needed by the vector store implementation
        # vector_store.persist() # Chroma usually persists automatically to configured dir

        logger.info(f"Successfully indexed {len(split_docs)} chunks from {filename} into collection '{collection_name}'.")

    except Exception as e:
        logger.error(f"Failed to process or index document {filename}: {e}", exc_info=True)
        # Consider adding more specific error handling or cleanup
    finally:
        # Clean up the temporary file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.debug(f"Removed temporary file: {file_path}")
            except OSError as e:
                logger.error(f"Error removing temporary file {file_path}: {e}")


# --- API Endpoints ---

@router.post(
    "/documents/index",
    response_model=IndexResponse,
    status_code=status.HTTP_202_ACCEPTED, # Accepted for background processing
    summary="Upload and index a document",
    dependencies=[Depends(get_api_key)]
)
async def index_document(
    background_tasks: BackgroundTasks,
    vector_store: Chroma = Depends(get_vector_store),
    embeddings: Embeddings = Depends(get_embeddings),
    s3_client = Depends(get_s3_client), # Inject S3 client
    file: UploadFile = File(...),
    collection_name: Optional[str] = Form("default"), # Example: index into collections
    metadata_json: Optional[str] = Form(None) # Pass metadata as JSON string
):
    """
    Accepts a file upload, saves it temporarily (or uploads to S3),
    and schedules background task for processing and indexing.
    """
    logger.info(f"Received request to index document: {file.filename} into collection '{collection_name}'")

    metadata = {}
    if metadata_json:
        import json
        try:
            metadata = json.loads(metadata_json)
            if not isinstance(metadata, dict):
                raise ValueError("Metadata must be a JSON object")
            logger.debug(f"Parsed metadata: {metadata}")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON provided for metadata.")
        except ValueError as e:
             raise HTTPException(status_code=400, detail=str(e))


    # Create a temporary directory for safe file handling
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)

    try:
        # Save uploaded file temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Temporarily saved uploaded file to: {temp_file_path}")

        # --- Option 1: Process locally then delete ---
        # Add the processing task to the background
        background_tasks.add_task(
            process_and_index_document,
            file_path=temp_file_path,
            filename=file.filename,
            vector_store=vector_store,
            embeddings=embeddings,
            collection_name=collection_name,
            metadata=metadata
            # Note: Pass copies or pickleable objects to background tasks
        )
        task_id = str(uuid4()) # Generate a task ID for tracking (if needed)
        logger.info(f"Scheduled background task '{task_id}' for indexing {file.filename}")

        # --- Option 2: Upload to S3 first, then process ---
        # if settings.S3_ENABLED and s3_client and settings.S3_BUCKET_NAME:
        #     s3_key = f"uploads/{collection_name}/{uuid4()}_{file.filename}"
        #     try:
        #         logger.info(f"Uploading {file.filename} to S3 bucket {settings.S3_BUCKET_NAME} with key {s3_key}")
        #         await upload_to_s3(s3_client, settings.S3_BUCKET_NAME, s3_key, temp_file_path)
        #         logger.info("Successfully uploaded to S3.")
        #         os.remove(temp_file_path) # Remove local temp file after S3 upload

        #         # Schedule processing using the S3 key
        #         # The background task would need to download from S3 first
        #         # background_tasks.add_task(process_s3_document, s3_key, ...)

        #     except Exception as e:
        #         logger.error(f"Failed to upload {file.filename} to S3: {e}", exc_info=True)
        #         # Cleanup local file even if S3 fails
        #         if os.path.exists(temp_file_path): os.remove(temp_file_path)
        #         if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        #         raise HTTPException(status_code=500, detail="Failed to upload file to storage.")
        # else:
        #     # Process locally if S3 is not enabled/configured (as done in Option 1)
        #     background_tasks.add_task(process_and_index_document, ...) # As above

        # --- Cleanup ---
        # Ensure the temp directory itself is cleaned up after the request,
        # not immediately, as the background task needs the file.
        # A more robust solution might involve the background task deleting its own file/dir.
        # For simplicity here, we rely on OS temp cleanup or task-based cleanup.
        # Consider adding `shutil.rmtree(temp_dir)` inside the background task's `finally` block.

        return IndexResponse(
            message="Document accepted for indexing.",
            filename=file.filename,
            collection=collection_name,
            task_id=task_id # Return task ID
            # Add s3_key if uploaded
        )

    except Exception as e:
        logger.error(f"Error handling file upload for {file.filename}: {e}", exc_info=True)
        # Clean up directory in case of early failure
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=f"Failed to process upload: {e}")
    finally:
        # Ensure file object is closed
        await file.close()


@router.post(
    "/documents/query",
    response_model=QueryResponse,
    summary="Query indexed documents (RAG)",
    dependencies=[Depends(get_api_key)]
)
async def query_documents(
    request_body: QueryRequest,
    vector_store: Chroma = Depends(get_vector_store),
    llm: BaseChatModel = Depends(get_chat_llm)
):
    """
    Performs a similarity search in the vector store and uses the results
    to answer a query with the LLM (Retrieval-Augmented Generation).
    """
    logger.info(f"Received query for collection '{request_body.collection_name}': {request_body.query}")

    try:
        # Create a retriever from the vector store
        # Adjust search_kwargs as needed (k = number of results)
        # Add filtering if querying specific collections or metadata
        search_kwargs = {'k': request_body.top_k}
        if request_body.filter:
            search_kwargs['filter'] = request_body.filter
            logger.debug(f"Applying filter: {request_body.filter}")

        # Note: Assuming vector_store is Chroma. If using abstraction, adapt.
        # You might need to pass collection_name to the retriever/vector_store
        # retriever = vector_store.as_retriever(
        #     search_kwargs=search_kwargs,
        #     collection_name=request_body.collection_name # If supported
        #     )
        retriever = vector_store.as_retriever(search_kwargs=search_kwargs)


        # --- Option 1: Basic RetrievalQA ---
        # Chain execution is synchronous in standard RetrievalQA
        # from fastapi.concurrency import run_in_threadpool
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", # Options: "stuff", "map_reduce", "refine", "map_rerank"
            retriever=retriever,
            return_source_documents=True, # Get source docs used for the answer
            chain_type_kwargs={"prompt": None} # Optional: customize prompt
        )

        logger.debug("Running QA chain...")
        # result = await run_in_threadpool(qa_chain.invoke, {"query": request_body.query})
        # Use invoke directly if async context allows sync calls, or if chain is async
        result = await qa_chain.ainvoke({"query": request_body.query}) # Use async version

        logger.info("QA chain finished.")

        source_docs = []
        if result.get("source_documents"):
            for doc in result["source_documents"]:
                source_docs.append(DocumentMetadata(
                    source=doc.metadata.get("source", "Unknown"),
                    metadata=doc.metadata,
                    # score=doc.metadata.get("score") # Include score if retriever provides it
                    page_content_preview=doc.page_content[:200] + "..." # Preview
                ))

        return QueryResponse(
            answer=result.get("result", "No answer found."),
            sources=source_docs
        )

        # --- Option 2: Manual Retrieval and Generation (More Control) ---
        # logger.debug("Performing similarity search...")
        # # Use async version if available
        # # docs = await retriever.aget_relevant_documents(request_body.query)
        # docs = retriever.get_relevant_documents(request_body.query)
        # logger.info(f"Retrieved {len(docs)} relevant documents.")

        # if not docs:
        #     # Optionally, still ask the LLM without context
        #     # ai_response = await llm.ainvoke([HumanMessage(content=request_body.query)])
        #     # return QueryResponse(answer=ai_response.content, sources=[])
        #     return QueryResponse(answer="No relevant documents found to answer the query.", sources=[])

        # # Prepare context for LLM
        # context = "\n\n".join([doc.page_content for doc in docs])
        # prompt_template = f"""Use the following pieces of context to answer the question at the end.
        # If you don't know the answer, just say that you don't know, don't try to make up an answer.

        # Context:
        # {context}

        # Question: {request_body.query}
        # Helpful Answer:"""
        # messages = [HumanMessage(content=prompt_template)]

        # logger.debug("Generating answer with LLM using retrieved context...")
        # ai_response = await llm.ainvoke(messages)
        # logger.info("LLM generation complete.")

        # source_docs_metadata = [DocumentMetadata.from_langchain_doc(doc) for doc in docs]

        # return QueryResponse(
        #     answer=ai_response.content,
        #     sources=source_docs_metadata
        # )

    except Exception as e:
        logger.error(f"Error during document query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to query documents.")