"""
Data Plane Proxy Routes

Proxies requests to /v1/data-plane/* to the API Gateway HTTP API.
This allows the ECS Router to forward Data Plane requests through the
existing CloudFront -> ALB -> Router path.
"""
import os
import aiohttp
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import Response

router = APIRouter(prefix="/v1/data-plane", tags=["Data Plane"])

# API Gateway configuration for Data Plane endpoints
# Use the public API Gateway URL - traffic goes via NAT gateway
DATA_PLANE_API_GATEWAY_URL = os.getenv(
    "DATA_PLANE_API_GATEWAY_URL",
    "https://qqf2ajs1b7.execute-api.us-east-1.amazonaws.com"
)


async def proxy_request(request: Request, path: str) -> Response:
    """
    Proxy a request to the Data Plane API Gateway.
    """
    # Build target URL
    target_url = f"{DATA_PLANE_API_GATEWAY_URL}/v1/data-plane/{path}"

    # Get request body
    body = await request.body()

    # Headers to explicitly forward (allowlist approach to avoid forwarding
    # CloudFront/ALB headers that cause API Gateway to reject requests)
    allowed_headers = {
        "content-type",
        "accept",
        "authorization",
        "x-api-key",
        "x-license-key",
        "user-agent",
    }

    # Build headers to forward to API Gateway
    headers = {}
    for key, value in request.headers.items():
        if key.lower() in allowed_headers:
            headers[key] = value

    # Use skip_auto_headers to have full control over headers sent
    async with aiohttp.ClientSession(skip_auto_headers=['User-Agent']) as session:
        try:
            async with session.request(
                method=request.method,
                url=target_url,
                headers=headers,
                data=body if body else None,
                timeout=aiohttp.ClientTimeout(total=30),
                ssl=True
            ) as resp:
                # Read response
                response_body = await resp.read()

                # Build response headers (exclude hop-by-hop headers)
                response_headers = {}
                for key, value in resp.headers.items():
                    key_lower = key.lower()
                    if key_lower not in ("transfer-encoding", "connection", "keep-alive"):
                        response_headers[key] = value

                return Response(
                    content=response_body,
                    status_code=resp.status,
                    headers=response_headers,
                    media_type=resp.headers.get("Content-Type", "application/json")
                )
        except aiohttp.ClientError as e:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to proxy request to Data Plane: {str(e)}"
            )


# License endpoints
@router.post("/license/validate")
async def license_validate(request: Request):
    """Proxy license validation to API Gateway."""
    return await proxy_request(request, "license/validate")


@router.post("/license")
async def license_create(request: Request):
    """Proxy license creation to API Gateway."""
    return await proxy_request(request, "license")


@router.get("/license")
async def license_list(request: Request):
    """Proxy license list to API Gateway."""
    return await proxy_request(request, "license")


@router.delete("/license/{license_id}")
async def license_delete(request: Request, license_id: str):
    """Proxy license deletion to API Gateway."""
    return await proxy_request(request, f"license/{license_id}")


# Config endpoints
@router.get("/config")
async def config_get(request: Request):
    """Proxy config retrieval to API Gateway."""
    return await proxy_request(request, "config")


@router.post("/config")
async def config_create(request: Request):
    """Proxy config creation to API Gateway."""
    return await proxy_request(request, "config")


@router.put("/config")
async def config_update(request: Request):
    """Proxy config update to API Gateway."""
    return await proxy_request(request, "config")


# Auth endpoints
@router.post("/auth")
async def auth_token(request: Request):
    """Proxy auth token request to API Gateway."""
    return await proxy_request(request, "auth")


# Telemetry endpoints
@router.post("/telemetry")
async def telemetry_ingest(request: Request):
    """Proxy telemetry ingest to API Gateway."""
    return await proxy_request(request, "telemetry")


@router.get("/telemetry")
async def telemetry_query(request: Request):
    """Proxy telemetry query to API Gateway."""
    return await proxy_request(request, "telemetry")


# Catch-all for any other data-plane paths
@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def catch_all(request: Request, path: str):
    """Catch-all proxy for any data-plane path not explicitly defined."""
    return await proxy_request(request, path)
