from fastapi import APIRouter, Query, Request, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import Optional
import time
import json

from ..database.ConversationHandler import ConversationHandler
from ..schemas import (
    ConversationCreate,
    ConversationUpdate,
    ConversationResponse,
    ConversationListResponse,
    MessageCreate,
    MessageResponse,
    ConversationMessagesResponse,
    SendMessageResponse,
    ChatMessage,
    Usage,
)
from ..helpers.utils import _parse_model_string, adapters_for
from ..router import HealthFirstPlanner
from ..database.PricingHandler import PricingHandler
from ..database.DeploymentHandler import DeploymentHandler

router = APIRouter(prefix="/v1/conversations", tags=["conversations"])
planner = HealthFirstPlanner()


# ============== Conversation CRUD ==============

@router.post("", response_model=ConversationResponse)
async def create_conversation(
    request: Request,
    body: ConversationCreate,
):
    """Create a new conversation"""
    tenant_id = getattr(request.state, 'tenant_id', None)
    if not tenant_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    conversation = await ConversationHandler.create_conversation(
        tenant_id=tenant_id,
        title=body.title,
        model=body.model,
        system_prompt=body.system_prompt,
        metadata=body.metadata
    )

    return ConversationResponse(**conversation)


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    request: Request,
    limit: int = Query(20, ge=1, le=100, description="Number of conversations to return"),
    cursor: Optional[str] = Query(None, description="Pagination cursor"),
):
    """List all conversations for the authenticated tenant"""
    tenant_id = getattr(request.state, 'tenant_id', None)
    if not tenant_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    conversations, next_cursor, total = await ConversationHandler.list_conversations(
        tenant_id=tenant_id,
        limit=limit,
        cursor=cursor
    )

    return ConversationListResponse(
        conversations=[ConversationResponse(**c) for c in conversations],
        total=total,
        has_more=next_cursor is not None,
        next_cursor=next_cursor
    )


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    request: Request,
    conversation_id: str,
):
    """Get a specific conversation"""
    tenant_id = getattr(request.state, 'tenant_id', None)
    if not tenant_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    conversation = await ConversationHandler.get_conversation(
        tenant_id=tenant_id,
        conversation_id=conversation_id
    )

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ConversationResponse(**conversation)


@router.patch("/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    request: Request,
    conversation_id: str,
    body: ConversationUpdate,
):
    """Update a conversation (title, model, etc.)"""
    tenant_id = getattr(request.state, 'tenant_id', None)
    if not tenant_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    conversation = await ConversationHandler.get_conversation(tenant_id, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    updated = await ConversationHandler.update_conversation(
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        title=body.title,
        model=body.model,
        system_prompt=body.system_prompt,
        metadata=body.metadata
    )

    return ConversationResponse(**updated)


@router.delete("/{conversation_id}")
async def delete_conversation(
    request: Request,
    conversation_id: str,
):
    """Delete a conversation and all its messages"""
    tenant_id = getattr(request.state, 'tenant_id', None)
    if not tenant_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    conversation = await ConversationHandler.get_conversation(tenant_id, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    await ConversationHandler.delete_conversation(tenant_id, conversation_id)

    return {"status": "deleted", "conversation_id": conversation_id}


# ============== Messages ==============

@router.get("/{conversation_id}/messages", response_model=ConversationMessagesResponse)
async def get_messages(
    request: Request,
    conversation_id: str,
    limit: int = Query(100, ge=1, le=500, description="Number of messages to return"),
    cursor: Optional[str] = Query(None, description="Pagination cursor"),
    order: str = Query("asc", description="Sort order: 'asc' (oldest first) or 'desc' (newest first)"),
):
    """Get messages for a conversation"""
    tenant_id = getattr(request.state, 'tenant_id', None)
    if not tenant_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    conversation = await ConversationHandler.get_conversation(tenant_id, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages, next_cursor = await ConversationHandler.get_messages(
        conversation_id=conversation_id,
        limit=limit,
        cursor=cursor,
        order=order
    )

    return ConversationMessagesResponse(
        conversation_id=conversation_id,
        messages=[MessageResponse(**m) for m in messages],
        total=conversation["message_count"],
        has_more=next_cursor is not None,
        next_cursor=next_cursor
    )


@router.post("/{conversation_id}/messages", response_model=SendMessageResponse)
async def send_message(
    request: Request,
    conversation_id: str,
    body: MessageCreate,
    background_tasks: BackgroundTasks,
):
    """
    Send a message in a conversation.

    This endpoint:
    1. Stores the user message
    2. Retrieves conversation history
    3. Calls the router for LLM inference
    4. Stores and returns the assistant response
    """
    tenant_id = getattr(request.state, 'tenant_id', None)
    if not tenant_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    conversation = await ConversationHandler.get_conversation(tenant_id, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    model = body.model or conversation["model"]
    logical_model, forced_provider = _parse_model_string(model)

    # 1. Store user message
    user_message = await ConversationHandler.add_message(
        conversation_id=conversation_id,
        tenant_id=tenant_id,
        role="user",
        content=body.content
    )

    # 2. Get conversation history for context
    history = await ConversationHandler.get_all_messages_for_context(conversation_id)

    # 3. Build messages list for LLM
    messages = [ChatMessage(role=m["role"], content=m["content"]) for m in history]

    # 4. Call router for inference
    adapters = await adapters_for(logical_model, forced_provider=forced_provider, tenant_id=tenant_id)

    if forced_provider:
        candidate = next((a for a in adapters if a.name == forced_provider), None)
        if not candidate:
            raise HTTPException(
                status_code=404,
                detail=f"Provider '{forced_provider}' not found for model '{logical_model}'"
            )
        ordered = [candidate]
    else:
        ordered = await planner.rank(logical_model, adapters)

    # Inference loop (with fallback)
    last_error = None
    resp_text = None
    chosen_adapter_name = None
    metrics_used = {}

    for chosen in ordered:
        resp, metrics = await chosen.send({
            "tenant": tenant_id,
            "model": logical_model,
            "messages": [m.model_dump() for m in messages],
            "stream": False,
        })

        if "error" not in resp:
            chosen_adapter_name = chosen.name
            resp_text = resp.get("text", "")
            metrics_used = metrics
            break

        last_error = resp.get("error")

    if resp_text is None:
        raise HTTPException(
            status_code=502,
            detail=f"All providers failed: {last_error}"
        )

    # Calculate costs
    prompt_tokens = int(metrics_used.get("tokens_in", 0))
    completion_tokens = int(metrics_used.get("tokens_out", 0))
    latency_ms = float(metrics_used.get("latency_ms", 0))

    cost_usd = None
    if chosen_adapter_name:
        if chosen_adapter_name == "pureai":
            # Check if adapter provided inference cost (new behavior)
            if "inference_cost_usd" in metrics_used:
                # Use inference-time pricing
                cost_usd = round(metrics_used.get("inference_cost_usd", 0.0), 10)
            else:
                # Fallback to token-based pricing (backwards compatibility)
                price = {"input_per_million": 0.10, "output_per_million": 0.10, "cache_input_per_million": 0.0}
                bd = PricingHandler.breakdown_usd(
                    price=price,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cached_prompt_tokens=0
                )
                cost_usd = round(bd["total_cost_usd"], 10)
        else:
            price_response = await PricingHandler.get_price(chosen_adapter_name, logical_model)
            price = price_response.dict() if price_response else {"input_per_million": 0, "output_per_million": 0, "cache_input_per_million": 0}

            bd = PricingHandler.breakdown_usd(
                price=price,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_prompt_tokens=0
            )
            cost_usd = round(bd["total_cost_usd"], 10)

    # 5. Store assistant message
    assistant_message = await ConversationHandler.add_message(
        conversation_id=conversation_id,
        tenant_id=tenant_id,
        role="assistant",
        content=resp_text,
        model=logical_model,
        provider=chosen_adapter_name,
        tokens_in=prompt_tokens,
        tokens_out=completion_tokens,
        latency_ms=latency_ms,
        cost_usd=cost_usd
    )

    # 6. Update conversation title if it's the first user message
    if conversation["message_count"] <= 1:
        title = body.content[:50] + ("..." if len(body.content) > 50 else "")
        await ConversationHandler.update_conversation(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            title=title
        )

    return SendMessageResponse(
        user_message=MessageResponse(**user_message),
        assistant_message=MessageResponse(**assistant_message),
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
            total_cost_usd=cost_usd
        )
    )


@router.post("/{conversation_id}/messages/stream")
async def send_message_stream(
    request: Request,
    conversation_id: str,
    body: MessageCreate,
    background_tasks: BackgroundTasks,
):
    """
    Send a message with streaming response.
    Returns Server-Sent Events stream.
    """
    tenant_id = getattr(request.state, 'tenant_id', None)
    if not tenant_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    conversation = await ConversationHandler.get_conversation(tenant_id, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    model = body.model or conversation["model"]
    logical_model, forced_provider = _parse_model_string(model)

    # Store user message
    user_message = await ConversationHandler.add_message(
        conversation_id=conversation_id,
        tenant_id=tenant_id,
        role="user",
        content=body.content
    )

    # Get history
    history = await ConversationHandler.get_all_messages_for_context(conversation_id)
    messages = [ChatMessage(role=m["role"], content=m["content"]) for m in history]

    # Get adapters
    adapters = await adapters_for(logical_model, forced_provider=forced_provider, tenant_id=tenant_id)

    if forced_provider:
        candidate = next((a for a in adapters if a.name == forced_provider), None)
        if not candidate:
            raise HTTPException(status_code=404, detail=f"Provider not found")
        ordered = [candidate]
    else:
        ordered = await planner.rank(logical_model, adapters)

    async def stream_with_storage():
        """Generate stream and store final message"""
        collected_text = ""
        start_time = time.time()
        chosen_adapter = None
        last_error = None

        for chosen in ordered:
            resp, metrics = await chosen.send({
                "tenant": tenant_id,
                "model": logical_model,
                "messages": [m.model_dump() for m in messages],
                "stream": True,
            })

            if "error" not in resp and "stream" in resp:
                chosen_adapter = chosen

                async for chunk in resp["stream"]:
                    try:
                        delta_content = chunk.choices[0].delta.get("content", "") or ""
                    except Exception:
                        delta_content = getattr(getattr(chunk.choices[0], "delta", None), "content", "") or ""

                    if delta_content:
                        collected_text += delta_content
                        chunk_data = {
                            "type": "content",
                            "content": delta_content
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"

                break
            else:
                # Store the error for reporting
                last_error = resp.get("error", "Unknown error")

        # Calculate final metrics
        latency_ms = (time.time() - start_time) * 1000

        # If no adapter succeeded, send error to frontend
        if not chosen_adapter and last_error:
            error_event = {
                "type": "error",
                "error": str(last_error)
            }
            yield f"data: {json.dumps(error_event)}\n\n"

            # Store error as assistant message
            assistant_msg = await ConversationHandler.add_message(
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                role="assistant",
                content=f"Error: {last_error}",
                model=logical_model,
                provider=None,
                latency_ms=latency_ms,
                tokens_out=0
            )

            final_event = {
                "type": "done",
                "message_id": assistant_msg["message_id"],
                "latency_ms": latency_ms,
                "error": True
            }
            yield f"data: {json.dumps(final_event)}\n\n"
            return

        tokens_out = max(5, int(len(collected_text.split()) * 1.3))

        # Store assistant message
        assistant_msg = await ConversationHandler.add_message(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            role="assistant",
            content=collected_text,
            model=logical_model,
            provider=chosen_adapter.name if chosen_adapter else None,
            latency_ms=latency_ms,
            tokens_out=tokens_out
        )

        # Update title if first message
        conv = await ConversationHandler.get_conversation(tenant_id, conversation_id)
        if conv and conv["message_count"] <= 2:
            title = body.content[:50] + ("..." if len(body.content) > 50 else "")
            await ConversationHandler.update_conversation(
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                title=title
            )

        # Send final event with metadata
        final_event = {
            "type": "done",
            "message_id": assistant_msg["message_id"],
            "latency_ms": latency_ms
        }
        yield f"data: {json.dumps(final_event)}\n\n"

    return StreamingResponse(
        stream_with_storage(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )
