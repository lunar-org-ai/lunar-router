"""Error type classifications for router traces."""

from enum import Enum


class ErrorCategory(str, Enum):
    """Categories of errors that can occur during LLM provider calls."""

    AUTH_ERROR = "auth_error"  # 401, API key invalid, authentication failed
    RATE_LIMIT = "rate_limit"  # 429, quota exceeded, too many requests
    TIMEOUT = "timeout"  # Request timeout, connection timeout
    NETWORK_ERROR = "network_error"  # Connection failed, DNS error
    MODEL_ERROR = "model_error"  # Model unavailable, context too long, invalid model
    DEPLOYMENT_ERROR = "deployment_error"  # PureAI: deployment paused/starting/not found
    STREAMING_ERROR = "streaming_error"  # Error during streaming response
    CONTENT_FILTER = "content_filter"  # Content blocked by safety filter
    INVALID_REQUEST = "invalid_request"  # 400, bad request, invalid parameters
    SERVER_ERROR = "server_error"  # 500, 502, 503, provider server error
    UNKNOWN = "unknown"  # Unclassified error
