"""
Error classification for routing decisions.

Standardized error categories allow the router to make intelligent
retry/skip decisions based on error type.
"""

from enum import Enum
from typing import Optional, Union
import re


class ErrorCategory(str, Enum):
    """Standardized error categories for routing."""

    AUTH_ERROR = "auth_error"  # API key/permission issues
    RATE_LIMIT = "rate_limit"  # Quota/throttling
    TIMEOUT = "timeout"  # Deadline exceeded
    NETWORK_ERROR = "network_error"  # Connection issues
    CONTENT_FILTER = "content_filter"  # Safety/moderation blocks
    MODEL_ERROR = "model_error"  # Context limits, model not found
    DEPLOYMENT_ERROR = "deployment_error"  # Self-hosted deployments (paused, scaling)
    STREAMING_ERROR = "streaming_error"  # Stream protocol issues
    INVALID_REQUEST = "invalid_request"  # Malformed requests
    SERVER_ERROR = "server_error"  # 5xx errors
    UNKNOWN = "unknown"  # Unclassified


# Patterns for error message classification
_AUTH_PATTERNS = [
    r"invalid.*api.*key",
    r"unauthorized",
    r"authentication",
    r"permission denied",
    r"access denied",
    r"invalid.*token",
    r"api_key",
]

_RATE_LIMIT_PATTERNS = [
    r"rate.?limit",
    r"too many requests",
    r"quota",
    r"throttl",
    r"capacity",
    r"overloaded",
    r"429",
]

_TIMEOUT_PATTERNS = [
    r"timeout",
    r"timed out",
    r"deadline exceeded",
    r"request took too long",
]

_NETWORK_PATTERNS = [
    r"connection.*refused",
    r"connection.*reset",
    r"connection.*error",
    r"network.*error",
    r"dns.*error",
    r"socket.*error",
    r"ssl.*error",
    r"certificate.*error",
]

_CONTENT_FILTER_PATTERNS = [
    r"content.*filter",
    r"safety",
    r"moderation",
    r"harmful",
    r"inappropriate",
    r"violat",
    r"blocked.*content",
    r"content.*policy",
]

_MODEL_ERROR_PATTERNS = [
    r"context.*length",
    r"max.*tokens",
    r"model.*not.*found",
    r"model.*unavailable",
    r"invalid.*model",
    r"unsupported.*model",
    r"token.*limit",
]

_DEPLOYMENT_PATTERNS = [
    r"deployment.*paused",
    r"scaling",
    r"endpoint.*not.*found",
    r"sagemaker",
    r"inference.*endpoint",
]

_STREAMING_PATTERNS = [
    r"stream.*error",
    r"sse.*error",
    r"chunk.*error",
]

_INVALID_REQUEST_PATTERNS = [
    r"invalid.*request",
    r"bad.*request",
    r"malformed",
    r"missing.*parameter",
    r"invalid.*parameter",
    r"validation.*error",
    r"400",
]

_SERVER_ERROR_PATTERNS = [
    r"internal.*server.*error",
    r"server.*error",
    r"500",
    r"502",
    r"503",
    r"504",
]


def classify_error(
    error: Union[Exception, str, None],
    status_code: Optional[int] = None,
) -> ErrorCategory:
    """
    Classify an error into a standard category.

    Args:
        error: Exception, error message string, or None
        status_code: Optional HTTP status code

    Returns:
        ErrorCategory enum value
    """
    if error is None:
        return ErrorCategory.UNKNOWN

    # Convert to lowercase string for pattern matching
    if isinstance(error, Exception):
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
    else:
        error_str = str(error).lower()
        error_type = ""

    # Check status code first
    if status_code:
        if status_code == 401 or status_code == 403:
            return ErrorCategory.AUTH_ERROR
        if status_code == 429:
            return ErrorCategory.RATE_LIMIT
        if status_code == 408:
            return ErrorCategory.TIMEOUT
        if status_code == 400:
            # Could be invalid request or content filter
            if _matches_patterns(error_str, _CONTENT_FILTER_PATTERNS):
                return ErrorCategory.CONTENT_FILTER
            return ErrorCategory.INVALID_REQUEST
        if 500 <= status_code < 600:
            return ErrorCategory.SERVER_ERROR

    # Check exception type
    if "timeout" in error_type:
        return ErrorCategory.TIMEOUT
    if "connection" in error_type:
        return ErrorCategory.NETWORK_ERROR
    if "auth" in error_type:
        return ErrorCategory.AUTH_ERROR

    # Pattern matching on error message
    pattern_checks = [
        (_AUTH_PATTERNS, ErrorCategory.AUTH_ERROR),
        (_RATE_LIMIT_PATTERNS, ErrorCategory.RATE_LIMIT),
        (_TIMEOUT_PATTERNS, ErrorCategory.TIMEOUT),
        (_NETWORK_PATTERNS, ErrorCategory.NETWORK_ERROR),
        (_CONTENT_FILTER_PATTERNS, ErrorCategory.CONTENT_FILTER),
        (_MODEL_ERROR_PATTERNS, ErrorCategory.MODEL_ERROR),
        (_DEPLOYMENT_PATTERNS, ErrorCategory.DEPLOYMENT_ERROR),
        (_STREAMING_PATTERNS, ErrorCategory.STREAMING_ERROR),
        (_INVALID_REQUEST_PATTERNS, ErrorCategory.INVALID_REQUEST),
        (_SERVER_ERROR_PATTERNS, ErrorCategory.SERVER_ERROR),
    ]

    for patterns, category in pattern_checks:
        if _matches_patterns(error_str, patterns):
            return category

    return ErrorCategory.UNKNOWN


def _matches_patterns(text: str, patterns: list) -> bool:
    """Check if text matches any of the given regex patterns."""
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def is_retryable(category: ErrorCategory) -> bool:
    """
    Determine if an error category is retryable.

    Args:
        category: Error category

    Returns:
        True if the error is potentially transient and retryable
    """
    retryable = {
        ErrorCategory.RATE_LIMIT,
        ErrorCategory.TIMEOUT,
        ErrorCategory.NETWORK_ERROR,
        ErrorCategory.SERVER_ERROR,
        ErrorCategory.DEPLOYMENT_ERROR,
        ErrorCategory.STREAMING_ERROR,
    }
    return category in retryable


def should_skip_provider(category: ErrorCategory) -> bool:
    """
    Determine if an error should cause the provider to be skipped.

    Args:
        category: Error category

    Returns:
        True if the provider should be skipped for future requests
    """
    skip = {
        ErrorCategory.AUTH_ERROR,  # Bad credentials - won't work
        ErrorCategory.MODEL_ERROR,  # Model not supported
        ErrorCategory.DEPLOYMENT_ERROR,  # Deployment issue
    }
    return category in skip
