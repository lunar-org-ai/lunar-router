"""
Pydantic schemas for router-core.

OpenAI-compatible request/response models.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


# Chat Completion schemas (OpenAI-compatible)

class ChatMessage(BaseModel):
    """A single message in a chat conversation."""
    role: str
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    user: Optional[str] = None
    # Provider routing hints
    provider: Optional[str] = None  # Force specific provider


class ChatChoice(BaseModel):
    """A single choice in chat completion response."""
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    # Cost breakdown (router extension)
    input_cost_usd: Optional[float] = None
    output_cost_usd: Optional[float] = None
    cache_input_cost_usd: Optional[float] = None
    total_cost_usd: Optional[float] = None
    # Performance metrics (router extension)
    latency_ms: Optional[float] = None
    ttft_ms: Optional[float] = None


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Optional[Usage] = None
    # Router metadata (extension)
    provider: Optional[str] = None


# Streaming schemas

class DeltaContent(BaseModel):
    """Delta content for streaming."""
    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(BaseModel):
    """Streaming choice."""
    index: int
    delta: DeltaContent
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """Streaming chunk for chat completion."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]


# Completions (legacy/text) schemas

class CompletionRequest(BaseModel):
    """Legacy completion request."""
    model: str
    prompt: str
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    stop: Optional[List[str]] = None
    stream: Optional[bool] = False


class CompletionChoice(BaseModel):
    """Completion choice."""
    index: int
    text: str
    finish_reason: str = "stop"


class CompletionResponse(BaseModel):
    """Legacy completion response."""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Optional[Usage] = None


# Models list (OpenAI-compatible /v1/models)

class ModelProviderInfo(BaseModel):
    """Provider info for a model."""
    id: str
    type: str = "llm"
    enabled: bool = True


class ModelInfo(BaseModel):
    """Model information."""
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "router"
    providers: List[ModelProviderInfo] = []


class ModelsListResponse(BaseModel):
    """Response for /v1/models endpoint."""
    object: str = "list"
    data: List[ModelInfo]


# Error response

class ErrorDetail(BaseModel):
    """Error detail."""
    message: str
    type: str = "invalid_request_error"
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response."""
    error: ErrorDetail


# Router-specific schemas

class InferRequest(BaseModel):
    """Router inference request."""
    model: str
    prompt: str
    constraints: Optional[Dict[str, Any]] = None
    stream: Optional[bool] = False


class InferResponse(BaseModel):
    """Router inference response."""
    provider: str
    model: str
    text: str
    metrics: Dict[str, float]
    chosen_by: Dict[str, Any]


class ProviderInfo(BaseModel):
    """Provider information."""
    id: str
    type: str
    enabled: bool
    params: Dict[str, Any] = {}


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    providers: Dict[str, Dict[str, Any]] = {}
