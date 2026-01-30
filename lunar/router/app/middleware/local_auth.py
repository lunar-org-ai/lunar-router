"""
Simple local authentication middleware for development.
Uses environment-based API keys instead of AWS Cognito/Lambda.
"""
import os
import hashlib
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Local API key configuration
# In production, use proper secrets management
LOCAL_API_KEYS: Dict[str, Dict[str, Any]] = {}


def _load_api_keys():
    """Load API keys from environment or config file."""
    global LOCAL_API_KEYS

    # Load from environment variable (comma-separated: key1:tenant1,key2:tenant2)
    keys_env = os.getenv("LUNAR_API_KEYS", "")
    if keys_env:
        for entry in keys_env.split(","):
            if ":" in entry:
                key, tenant = entry.strip().split(":", 1)
                key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
                LOCAL_API_KEYS[key] = {
                    "tenant_id": tenant,
                    "tenant_name": tenant,
                    "key_hash": key_hash
                }

    # Default development key if none configured
    if not LOCAL_API_KEYS:
        LOCAL_API_KEYS["lunar-dev-key"] = {
            "tenant_id": "dev",
            "tenant_name": "Development",
            "key_hash": "dev-hash"
        }
        logger.warning("No API keys configured. Using default development key: 'lunar-dev-key'")


# Load keys on module import
_load_api_keys()


class LocalAuthMiddleware(BaseHTTPMiddleware):
    """Simple local authentication middleware."""

    def __init__(self, app):
        super().__init__(app)

        # Routes that do not require authentication
        self.exempt_paths = {
            "/health",
            "/docs",
            "/openapi.json",
            "/v1/pricing/models",
        }

        # Prefixes that do not require authentication
        self.exempt_prefixes = [
            "/v1/pricing/",
        ]

    def _is_exempt_prefix(self, path: str) -> bool:
        for prefix in self.exempt_prefixes:
            if path.startswith(prefix):
                return True
        return False

    async def dispatch(self, request: Request, call_next):
        # Allow CORS preflight requests
        if request.method == "OPTIONS":
            return await call_next(request)

        # Check exempt paths
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        if self._is_exempt_prefix(request.url.path):
            return await call_next(request)

        # Check for API key
        api_key = request.headers.get("x-api-key") or request.headers.get("Authorization", "").replace("Bearer ", "")

        if not api_key:
            # Allow requests without auth in dev mode
            dev_mode = os.getenv("LUNAR_DEV_MODE", "true").lower() == "true"
            if dev_mode:
                request.state.tenant_id = "dev"
                request.state.tenant_name = "Development"
                request.state.auth_method = "dev_mode"
                return await call_next(request)

            return JSONResponse(
                status_code=401,
                content={"detail": "Missing x-api-key header"}
            )

        # Validate API key
        key_info = LOCAL_API_KEYS.get(api_key)
        if not key_info:
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid API key"}
            )

        # Set request state
        request.state.tenant_id = key_info["tenant_id"]
        request.state.tenant_name = key_info["tenant_name"]
        request.state.auth_method = "api_key"
        request.state.key_hash = key_info["key_hash"]

        logger.debug(f"Authenticated request for tenant: {request.state.tenant_id}")

        return await call_next(request)
