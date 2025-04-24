from fastapi import APIRouter, Depends, HTTPException, Body, Request
from fastapi.responses import StreamingResponse
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from loguru import logger
from typing import List, AsyncGenerator

from app.models.schemas import ChatRequest, ChatResponse, StreamingChatResponse
from app.dependencies import get_chat_llm
from app.security.auth import get_api_key # Or your preferred auth dependency

router = APIRouter()

@router.post(
    "/chat/completion",
    response_model=ChatResponse,
    summary="Get a chat completion",
    dependencies=[Depends(get_api_key)] # Add authentication
)
async def chat_completion(
    request_body: ChatRequest,
    llm: BaseChatModel = Depends(get_chat_llm)
):
    """
    Receives a list of messages and returns a single completion from the configured LLM.
    """
    logger.info(f"Received chat completion request with {len(request_body.messages)} messages.")
    messages = []
    for msg in request_body.messages:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            messages.append(AIMessage(content=msg.content))
        elif msg.role == "system":
            messages.append(SystemMessage(content=msg.content))
        else:
            logger.warning(f"Unknown message role: {msg.role}")
            # Decide how to handle unknown roles, maybe raise error or ignore
            # raise HTTPException(status_code=400, detail=f"Invalid message role: {msg.role}")

    try:
        # Use ainvoke for async execution if the LLM supports it
        # Some LLMs/integrations might still be sync under the hood
        logger.debug(f"Sending messages to LLM: {messages}")
        ai_response = await llm.ainvoke(messages)
        logger.info("Received response from LLM.")
        return ChatResponse(role="assistant", content=ai_response.content)

    except Exception as e:
        logger.error(f"Error during chat completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get completion from LLM.")

@router.post(
    "/chat/stream",
    response_model=None, # Response is handled by StreamingResponse
    summary="Get a streaming chat completion",
    dependencies=[Depends(get_api_key)] # Add authentication
)
async def stream_chat_completion(
    request_body: ChatRequest,
    llm: BaseChatModel = Depends(get_chat_llm)
) -> StreamingResponse:
    """
    Receives a list of messages and streams the LLM's response chunk by chunk.
    Uses Server-Sent Events (SSE).
    """
    logger.info(f"Received streaming chat request with {len(request_body.messages)} messages.")
    messages = []
    for msg in request_body.messages:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            messages.append(AIMessage(content=msg.content))
        elif msg.role == "system":
            messages.append(SystemMessage(content=msg.content))

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # Use astream for async streaming
            logger.debug("Starting LLM stream...")
            async for chunk in llm.astream(messages):
                content = chunk.content
                if content: # Ensure content exists before sending
                    # SSE format: data: {"content": "...", "role": "assistant"}\n\n
                    response_chunk = StreamingChatResponse(role="assistant", content=content)
                    # Yield in SSE format
                    yield f"data: {response_chunk.json()}\n\n"
            logger.info("LLM stream finished.")
            # Optionally send a final 'done' message
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Error during streaming chat completion: {e}", exc_info=True)
            # Send an error message via SSE if possible, or just log
            error_message = StreamingChatResponse(role="error", content=f"Error: {e}")
            yield f"data: {error_message.json()}\n\n"
            # Need to decide if the connection should close or keep trying

    return StreamingResponse(event_generator(), media_type="text/event-stream")