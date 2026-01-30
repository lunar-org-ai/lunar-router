import json
import boto3
import jwt
from jwt import PyJWKClient
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Dict, Any, Optional
import os
import logging
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

# Cognito configuration
COGNITO_REGION = os.getenv("AWS_REGION", "us-east-1")
COGNITO_USER_POOL_ID = os.getenv("COGNITO_USER_POOL_ID", "us-east-1_Y147akc25")
COGNITO_ISSUER = f"https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{COGNITO_USER_POOL_ID}"
COGNITO_JWKS_URL = f"{COGNITO_ISSUER}/.well-known/jwks.json"

# Database configuration
DB_SECRET_ARN = os.getenv("DB_SECRET_ARN")

# Cache for JWKS client and DB connection
_jwks_client: Optional[PyJWKClient] = None
_db_connection = None
_db_secrets = None


def _get_jwks_client() -> PyJWKClient:
    """Get or create cached JWKS client."""
    global _jwks_client
    if _jwks_client is None:
        _jwks_client = PyJWKClient(COGNITO_JWKS_URL)
    return _jwks_client


def _get_db_secrets() -> Dict[str, Any]:
    """Fetch DB credentials from Secrets Manager."""
    global _db_secrets
    if _db_secrets is not None:
        return _db_secrets

    if not DB_SECRET_ARN:
        raise RuntimeError("DB_SECRET_ARN environment variable not set")

    sm = boto3.client("secretsmanager", region_name=COGNITO_REGION)
    resp = sm.get_secret_value(SecretId=DB_SECRET_ARN)
    secrets = json.loads(resp["SecretString"])

    _db_secrets = {
        "host": secrets.get("host"),
        "port": int(secrets.get("port", 5432)),
        "dbname": secrets.get("dbname") or secrets.get("dbName"),
        "user": secrets.get("user") or secrets.get("username"),
        "password": secrets.get("password"),
    }
    return _db_secrets


def _get_db_connection():
    """Get or create DB connection."""
    global _db_connection
    if _db_connection and getattr(_db_connection, "closed", 1) == 0:
        try:
            with _db_connection.cursor() as c:
                c.execute("SELECT 1;")
            return _db_connection
        except Exception:
            try:
                _db_connection.close()
            except Exception:
                pass
            _db_connection = None

    secrets = _get_db_secrets()
    _db_connection = psycopg2.connect(
        **secrets,
        connect_timeout=5,
        cursor_factory=RealDictCursor,
    )
    return _db_connection


def _get_tenant_id_from_cognito_sub(cognito_sub: str) -> Optional[str]:
    """Query RDS for tenant_id based on Cognito sub."""
    try:
        conn = _get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT tenant_id FROM users WHERE cognito_sub = %s LIMIT 1;",
                (cognito_sub,),
            )
            row = cur.fetchone()
            if row and row.get("tenant_id"):
                return str(row["tenant_id"])
            return None
    except Exception as e:
        logger.error(f"Failed to query tenant_id: {e}")
        return None


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, lambda_function_name: str = None):
        super().__init__(app)
        self.lambda_client = boto3.client('lambda')
        self.lambda_function_name = lambda_function_name or os.getenv(
            'AUTHORIZER_LAMBDA_NAME',
            'pureai-autodestill-dev-authorizer'
        )

        # Routes that do not require authentication
        self.exempt_paths = {
            "/health",
            "/docs",
            "/openapi.json",
            "/v1/pricing/models",  # Model discovery - public (from PricingTable)
        }

        # Prefixes that do not require authentication
        self.exempt_prefixes = [
            "/v1/pricing/",  # Pricing endpoints - public
            "/v1/data-plane/",  # Data Plane endpoints - use license key auth
        ]

        # Routes that use JWT authentication (Cognito Bearer token)
        self.jwt_auth_prefixes = [
            "/v1/stats",
            "/v1/conversations",
        ]

        # Routes that accept EITHER API key OR JWT (flexible auth)
        self.flexible_auth_prefixes = [
            "/v1/chat",
            "/v1/completions",
        ]

    def _is_jwt_auth_route(self, path: str) -> bool:
        """Check if the route should use JWT authentication."""
        for prefix in self.jwt_auth_prefixes:
            if path.startswith(prefix):
                return True
        return False

    def _is_flexible_auth_route(self, path: str) -> bool:
        """Check if the route accepts either API key or JWT."""
        for prefix in self.flexible_auth_prefixes:
            if path.startswith(prefix):
                return True
        return False

    def _is_exempt_prefix(self, path: str) -> bool:
        """Check if the route matches an exempt prefix."""
        for prefix in self.exempt_prefixes:
            if path.startswith(prefix):
                return True
        return False

    async def dispatch(self, request: Request, call_next):
        # Allow CORS preflight requests to pass through without authentication
        if request.method == "OPTIONS":
            return await call_next(request)

        # Verify if the path is exempted from authentication
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        # Verify if the path matches an exempt prefix
        if self._is_exempt_prefix(request.url.path):
            return await call_next(request)

        # Determine auth method based on route
        if self._is_flexible_auth_route(request.url.path):
            return await self._handle_flexible_auth(request, call_next)
        elif self._is_jwt_auth_route(request.url.path):
            return await self._handle_jwt_auth(request, call_next)
        else:
            return await self._handle_api_key_auth(request, call_next)

    async def _handle_jwt_auth(self, request: Request, call_next):
        """Handle JWT (Cognito Bearer token) authentication for stats routes."""
        auth_header = request.headers.get("Authorization")

        if not auth_header:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing Authorization header"}
            )

        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid Authorization header format. Expected 'Bearer <token>'"}
            )

        token = auth_header[7:]  # Remove "Bearer " prefix

        try:
            # Verify JWT token
            jwks_client = _get_jwks_client()
            signing_key = jwks_client.get_signing_key_from_jwt(token)

            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                issuer=COGNITO_ISSUER,
                options={"verify_aud": False}  # Cognito access tokens don't have aud
            )

            # Extract user info from claims
            cognito_sub = claims.get("sub") or claims.get("username")

            if not cognito_sub:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid token: missing sub claim"}
                )

            # Get tenant_id from database
            tenant_id = _get_tenant_id_from_cognito_sub(cognito_sub)

            if not tenant_id:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "User not associated with a tenant"}
                )

            # Set request state
            request.state.tenant_id = tenant_id
            request.state.cognito_sub = cognito_sub
            request.state.auth_method = "jwt"

            logger.info(f"JWT authenticated request for tenant: {tenant_id}")

        except jwt.ExpiredSignatureError:
            return JSONResponse(
                status_code=401,
                content={"detail": "Token has expired"}
            )
        except jwt.InvalidTokenError as e:
            logger.error(f"JWT validation error: {e}")
            return JSONResponse(
                status_code=401,
                content={"detail": f"Invalid token: {str(e)}"}
            )
        except Exception as e:
            logger.error(f"JWT auth error: {e}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Authentication service error"}
            )

        return await call_next(request)

    async def _handle_api_key_auth(self, request: Request, call_next):
        """Handle API key authentication for inference routes."""
        api_key = request.headers.get("x-api-key")
        if not api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing x-api-key header"}
            )

        try:
            auth_result = await self._invoke_authorizer(request, api_key)
            is_authorized = auth_result.get("isAuthorized", False)

            if not is_authorized:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Unauthorized"}
                )

            context = auth_result.get("context", {})

            request.state.tenant_id = context.get("tenant_id")
            request.state.tenant_name = context.get("tenant_name")
            request.state.auth_method = context.get("auth_method", "api_key")
            request.state.key_hash = context.get("key_hash")

            logger.info(f"API key authenticated request for tenant: {request.state.tenant_id}")

        except Exception as e:
            logger.error(f"Authorization error: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Authorization service error"}
            )

        return await call_next(request)

    async def _handle_flexible_auth(self, request: Request, call_next):
        """Handle flexible authentication - accepts either API key OR JWT."""
        api_key = request.headers.get("x-api-key")
        auth_header = request.headers.get("Authorization")

        # Try API key first if provided
        if api_key:
            try:
                auth_result = await self._invoke_authorizer(request, api_key)
                is_authorized = auth_result.get("isAuthorized", False)

                if is_authorized:
                    context = auth_result.get("context", {})
                    request.state.tenant_id = context.get("tenant_id")
                    request.state.tenant_name = context.get("tenant_name")
                    request.state.auth_method = "api_key"
                    request.state.key_hash = context.get("key_hash")
                    logger.info(f"Flexible auth (API key) for tenant: {request.state.tenant_id}")
                    return await call_next(request)
            except Exception as e:
                logger.warning(f"API key auth failed, will try JWT: {e}")

        # Try JWT if provided
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            try:
                jwks_client = _get_jwks_client()
                signing_key = jwks_client.get_signing_key_from_jwt(token)

                claims = jwt.decode(
                    token,
                    signing_key.key,
                    algorithms=["RS256"],
                    issuer=COGNITO_ISSUER,
                    options={"verify_aud": False}
                )

                cognito_sub = claims.get("sub") or claims.get("username")
                if cognito_sub:
                    tenant_id = _get_tenant_id_from_cognito_sub(cognito_sub)
                    if tenant_id:
                        request.state.tenant_id = tenant_id
                        request.state.cognito_sub = cognito_sub
                        request.state.auth_method = "jwt"
                        logger.info(f"Flexible auth (JWT) for tenant: {tenant_id}")
                        return await call_next(request)
            except Exception as e:
                logger.warning(f"JWT auth failed: {e}")

        # Neither worked
        return JSONResponse(
            status_code=401,
            content={"detail": "Authentication required. Provide either 'x-api-key' header or 'Authorization: Bearer <token>'"}
        )

    async def _invoke_authorizer(self, request: Request, api_key: str) -> Dict[str, Any]:
        payload = {
            "version": "2.0",
            "type": "REQUEST",
            "routeKey": f"{request.method} {request.url.path}",
            "headers": {
                "x-api-key": api_key
            }
        }

        try:
            response = self.lambda_client.invoke(
                FunctionName=self.lambda_function_name,
                InvocationType='RequestResponse',
                Payload=json.dumps(payload)
            )

            response_payload = json.loads(response['Payload'].read())
            logger.debug(f"Authorizer response: {response_payload}")
            return response_payload

        except Exception as e:
            logger.error(f"Failed to invoke authorizer lambda: {str(e)}")
            raise Exception(f"Authorizer invocation failed: {str(e)}")