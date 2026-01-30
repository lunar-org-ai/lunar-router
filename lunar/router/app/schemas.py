# app/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class InferBody(BaseModel):
    prompt: str
    constraints: Optional[Dict[str, Any]] = None
    stream: Optional[bool] = False

class InferRequest(BaseModel):
    model: str
    prompt: str
    constraints: Optional[Dict[str, Any]] = None
    stream: Optional[bool] = False

class InferResponse(BaseModel):
    provider: str
    model: str
    text: str
    metrics: Dict[str, float]
    chosen_by: Dict[str, Any]

class ProviderInfo(BaseModel):
    id: str
    type: str
    enabled: bool
    params: Dict[str, Any] = {}

#  OpenAI-style chat 
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    input_cost_usd: Optional[float] = None
    output_cost_usd: Optional[float] = None
    cache_input_cost_usd: Optional[float] = None
    total_cost_usd: Optional[float] = None
    latency_ms: Optional[float] = None
    ttft_ms: Optional[float] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    choices: List[ChatChoice]
    usage: Optional[Usage] = None


# Completions (legacy/text) schemas
class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    stop: Optional[List[str]] = None
    stream: Optional[bool] = False


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: str = "stop"


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    model: str
    choices: List[CompletionChoice]
    usage: Optional[Usage] = None


# Models list (OpenAI-compatible /v1/models)
class ModelProviderInfo(BaseModel):
    id: str
    type: str
    enabled: bool = True


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "pureai"
    providers: List[ModelProviderInfo] = []


class ModelsListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# Conversation schemas
class ConversationCreate(BaseModel):
    title: Optional[str] = None
    model: str = "gpt-4o-mini"
    system_prompt: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ConversationUpdate(BaseModel):
    title: Optional[str] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ConversationResponse(BaseModel):
    conversation_id: str
    tenant_id: str
    title: str
    model: str
    created_at: str
    updated_at: str
    message_count: int
    metadata: Optional[Dict[str, Any]] = None


class ConversationListResponse(BaseModel):
    conversations: List["ConversationResponse"]
    total: int
    has_more: bool
    next_cursor: Optional[str] = None


# Message schemas
class MessageCreate(BaseModel):
    content: str
    model: Optional[str] = None
    stream: Optional[bool] = False


class MessageResponse(BaseModel):
    message_id: str
    conversation_id: str
    role: str
    content: str
    created_at: str
    model: Optional[str] = None
    provider: Optional[str] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None


class ConversationMessagesResponse(BaseModel):
    conversation_id: str
    messages: List[MessageResponse]
    total: int
    has_more: bool
    next_cursor: Optional[str] = None


class SendMessageResponse(BaseModel):
    user_message: MessageResponse
    assistant_message: MessageResponse
    usage: Optional[Usage] = None
