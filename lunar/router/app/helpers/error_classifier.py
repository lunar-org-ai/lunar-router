"""Error classification utilities for router traces."""

from typing import Optional
from ..models.error_types import ErrorCategory


def classify_error(
    exception: Exception,
    response_status: Optional[int] = None,
    error_message: Optional[str] = None,
) -> str:
    """
    Classify an exception into an error category.

    Args:
        exception: The exception that was raised
        response_status: HTTP response status code if available
        error_message: Error message string if available

    Returns:
        ErrorCategory value as string
    """
    error_str = (error_message or str(exception)).lower()
    exc_type = type(exception).__name__

    # Check by status code first if available
    if response_status:
        if response_status == 401:
            return ErrorCategory.AUTH_ERROR.value
        elif response_status == 429:
            return ErrorCategory.RATE_LIMIT.value
        elif response_status == 400:
            return ErrorCategory.INVALID_REQUEST.value
        elif response_status >= 500:
            return ErrorCategory.SERVER_ERROR.value

    # Rate limiting patterns
    rate_limit_patterns = [
        "rate", "429", "quota", "too many requests", "throttl",
        "limit exceeded", "requests per", "tokens per"
    ]
    if any(p in error_str for p in rate_limit_patterns):
        return ErrorCategory.RATE_LIMIT.value

    # Auth errors
    auth_patterns = [
        "401", "auth", "api_key", "api key", "apikey", "unauthorized",
        "invalid key", "permission", "forbidden", "403", "access denied"
    ]
    if any(p in error_str for p in auth_patterns):
        return ErrorCategory.AUTH_ERROR.value

    # Timeout patterns
    timeout_patterns = ["timeout", "timed out", "deadline"]
    timeout_types = ("TimeoutError", "ConnectTimeout", "ReadTimeout", "WriteTimeout")
    if any(p in error_str for p in timeout_patterns) or exc_type in timeout_types:
        return ErrorCategory.TIMEOUT.value

    # Network errors
    network_patterns = ["connection", "network", "dns", "socket", "refused", "reset"]
    network_types = ("ConnectionError", "ConnectError", "ConnectionRefusedError")
    if any(p in error_str for p in network_patterns) or exc_type in network_types:
        return ErrorCategory.NETWORK_ERROR.value

    # Content filter
    content_patterns = [
        "content filter", "safety", "blocked", "moderation",
        "content policy", "harmful", "inappropriate"
    ]
    if any(p in error_str for p in content_patterns):
        return ErrorCategory.CONTENT_FILTER.value

    # Model errors
    model_patterns = [
        "context", "token limit", "max tokens", "context length",
        "model not found", "model_not_found", "does not exist",
        "invalid model", "not supported"
    ]
    if any(p in error_str for p in model_patterns):
        return ErrorCategory.MODEL_ERROR.value

    # Deployment errors (self-hosted models)
    deployment_patterns = [
        "deployment", "paused", "starting", "resuming", "not found",
        "vllm", "in_service", "scaling"
    ]
    if any(p in error_str for p in deployment_patterns):
        return ErrorCategory.DEPLOYMENT_ERROR.value

    # Streaming errors
    streaming_patterns = ["stream", "chunk", "sse", "event-stream"]
    if any(p in error_str for p in streaming_patterns):
        return ErrorCategory.STREAMING_ERROR.value

    # Invalid request
    invalid_patterns = ["400", "bad request", "invalid", "malformed", "missing required"]
    if any(p in error_str for p in invalid_patterns):
        return ErrorCategory.INVALID_REQUEST.value

    # Server errors
    server_patterns = ["500", "502", "503", "504", "internal server", "bad gateway", "unavailable"]
    if any(p in error_str for p in server_patterns):
        return ErrorCategory.SERVER_ERROR.value

    return ErrorCategory.UNKNOWN.value
