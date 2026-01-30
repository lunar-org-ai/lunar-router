from functools import lru_cache
import os
import json
import boto3
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
DEFAULT_BEDROCK_API_KEY = os.getenv("BEDROCK_API_KEY", "").strip()
DEFAULT_ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
DEFAULT_DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEFAULT_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
DEFAULT_MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "").strip()
DEFAULT_PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "").strip()
DEFAULT_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
DEFAULT_CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "").strip()
DEFAULT_COHERE_API_KEY = os.getenv("COHERE_API_KEY", "").strip()
DEFAULT_SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY", "").strip()

TENANT_SECRET_PREFIX = os.getenv(
    "TENANT_SECRET_PREFIX",
    "pureai/keys",
)

try:
    SECRETS_CLIENT = boto3.client("secretsmanager")
except Exception:
    SECRETS_CLIENT = None
    print(
        "[byok] WARNING: Failed to initialize Secrets Manager client; "
        "tenant-level BYOK resolution disabled.",
        flush=True,
    )


def is_byok_required(tenant_id: str) -> bool:
    """
   Defines whether the tenant requires strict BYOK.
    In the future:
    - read from RDS (tenants table),
    - or from token claims (e.g., `byok_required: true`).
    For now:
    - environment variable BYOK_REQUIRED_TENANTS="tenant1,tenant2,..."
    """
    if not tenant_id or tenant_id == "default":
        return False

    env_val = os.getenv("BYOK_REQUIRED_TENANTS", "")
    if not env_val:
        return False

    required_ids = {t.strip() for t in env_val.split(",") if t.strip()}
    return tenant_id in required_ids


@lru_cache(maxsize=256)
def _load_tenant_openai_key(tenant_id: str) -> Optional[str]:
    """
    Attempt to load the tenant-specific OpenAI key (BYOK) in Secrets Manager.
    No fallback applies here; only:
    - returns the key if found,
    - returns None if it doesn't exist/error.

    Checks both naming conventions:
    - {prefix}/{tenant_id}/openai (preferred)
    - {prefix}/{tenant_id}/openai_api_key (legacy)
    """
    if SECRETS_CLIENT is None:
        return None

    # Try both naming conventions
    secret_names = [
        f"{TENANT_SECRET_PREFIX}/{tenant_id}/openai",
        f"{TENANT_SECRET_PREFIX}/{tenant_id}/openai_api_key",
    ]

    for secret_name in secret_names:
        try:
            resp = SECRETS_CLIENT.get_secret_value(SecretId=secret_name)
            raw = resp.get("SecretString") or ""
            if not raw:
                continue

            try:
                data = json.loads(raw)
                key = (
                    data.get("api_key")
                    or data.get("OPENAI_API_KEY")
                    or data.get("openai_api_key")
                )
            except json.JSONDecodeError:
                key = raw

            if key:
                return key.strip()

        except SECRETS_CLIENT.exceptions.ResourceNotFoundException:
            continue
        except Exception as e:
            print(
                f"[byok] Failed to load OpenAI key for tenant='{tenant_id}' "
                f"(secret='{secret_name}'): {e}",
                flush=True,
            )

    return None


def get_key_for_tenant(tenant_id: str, byok_required: bool) -> str:
    """
    Strategy C — Mixed:

    - Empty or 'default' tenant:
    - always uses DEFAULT_OPENAI_API_KEY.
    - Managed tenant (byok_required=False):
    - tries BYOK (Secrets Manager);
    - if not found, falls back to the global key.
    - Byok_required tenant (byok_required=True):
    - tries BYOK;
    - if not found, ERROR (no fallback to the global key).
    """
    if not tenant_id or tenant_id == "default":
        return DEFAULT_OPENAI_API_KEY
    tenant_key = _load_tenant_openai_key(tenant_id)
    if tenant_key:
        return tenant_key

    if byok_required:
        raise RuntimeError(
            f"BYOK required for tenant '{tenant_id}' but no OpenAI key is configured"
        )

    return DEFAULT_OPENAI_API_KEY


@lru_cache(maxsize=256)
def _load_tenant_bedrock_credentials(tenant_id: str) -> Optional[dict]:
    """
    Attempt to load tenant-specific AWS IAM credentials for Bedrock from Secrets Manager.

    Supported formats:
    1. Direct format:
       {"aws_access_key_id": "AKIA...", "aws_secret_access_key": "...", "aws_region_name": "us-east-1"}

    2. Nested in api_key (from frontend):
       {"api_key": "{\"accessKey\":\"...\",\"secretKey\":\"...\",\"region\":\"...\"}"}

    3. Alternative field names:
       {"accessKey": "...", "secretKey": "...", "region": "..."}

    Checks both naming conventions:
    - {prefix}/{tenant_id}/bedrock (preferred)
    - {prefix}/{tenant_id}/bedrock_api_key (legacy)
    """
    if SECRETS_CLIENT is None:
        return None

    secret_names = [
        f"{TENANT_SECRET_PREFIX}/{tenant_id}/bedrock",
        f"{TENANT_SECRET_PREFIX}/{tenant_id}/bedrock_api_key",
    ]

    for secret_name in secret_names:
        try:
            resp = SECRETS_CLIENT.get_secret_value(SecretId=secret_name)
            raw = resp.get("SecretString") or ""
            if not raw:
                continue

            try:
                data = json.loads(raw)

                # Check if credentials are nested inside api_key field (frontend format)
                api_key_value = data.get("api_key")
                if api_key_value and isinstance(api_key_value, str):
                    try:
                        nested_data = json.loads(api_key_value)
                        if isinstance(nested_data, dict):
                            data = nested_data
                    except json.JSONDecodeError:
                        pass  # Not nested JSON, continue with original data

                # Extract credentials with multiple field name options
                access_key_id = (
                    data.get("aws_access_key_id") or data.get("access_key_id")
                    or data.get("accessKeyId") or data.get("accessKey")
                    or data.get("AWS_ACCESS_KEY_ID")
                )
                secret_access_key = (
                    data.get("aws_secret_access_key") or data.get("secret_access_key")
                    or data.get("secretAccessKey") or data.get("secretKey")
                    or data.get("AWS_SECRET_ACCESS_KEY")
                )
                region = (
                    data.get("aws_region_name") or data.get("region_name")
                    or data.get("region") or data.get("AWS_REGION") or "us-east-1"
                )

                if access_key_id and secret_access_key:
                    print(f"[byok] Loaded Bedrock IAM credentials for tenant='{tenant_id}', region='{region}'", flush=True)
                    return {
                        "aws_access_key_id": access_key_id.strip(),
                        "aws_secret_access_key": secret_access_key.strip(),
                        "aws_region_name": region.strip() if region else "us-east-1",
                    }

            except json.JSONDecodeError:
                print(f"[byok] Bedrock secret for tenant='{tenant_id}' is not valid JSON", flush=True)

        except SECRETS_CLIENT.exceptions.ResourceNotFoundException:
            continue
        except Exception as e:
            print(f"[byok] Failed to load Bedrock credentials for tenant='{tenant_id}': {e}", flush=True)

    return None


def get_bedrock_credentials_for_tenant(tenant_id: str, byok_required: bool) -> Optional[dict]:
    """
    Get AWS IAM credentials for Bedrock.

    Returns:
        Dict with aws_access_key_id, aws_secret_access_key, aws_region_name
        or None to use default credentials (IAM role / environment)
    """
    if not tenant_id or tenant_id == "default":
        print(f"[byok] Using default Bedrock credentials (IAM/environment) for tenant 'default'", flush=True)
        return None

    tenant_creds = _load_tenant_bedrock_credentials(tenant_id)
    if tenant_creds:
        return tenant_creds

    if byok_required:
        raise RuntimeError(
            f"BYOK required for tenant '{tenant_id}' but no Bedrock IAM credentials configured. "
            "Expected: {\"aws_access_key_id\": \"...\", \"aws_secret_access_key\": \"...\"}"
        )

    print(f"[byok] No Bedrock credentials for tenant '{tenant_id}', using default (IAM/environment)", flush=True)
    return None


@lru_cache(maxsize=256)
def _load_tenant_anthropic_key(tenant_id: str) -> Optional[str]:
    """
    Attempt to load the tenant-specific Anthropic API key (BYOK) in Secrets Manager.

    Checks both naming conventions:
    - {prefix}/{tenant_id}/anthropic (preferred)
    - {prefix}/{tenant_id}/anthropic_api_key (legacy)
    """
    if SECRETS_CLIENT is None:
        return None

    secret_names = [
        f"{TENANT_SECRET_PREFIX}/{tenant_id}/anthropic",
        f"{TENANT_SECRET_PREFIX}/{tenant_id}/anthropic_api_key",
    ]

    for secret_name in secret_names:
        try:
            resp = SECRETS_CLIENT.get_secret_value(SecretId=secret_name)
            raw = resp.get("SecretString") or ""
            if not raw:
                continue

            try:
                data = json.loads(raw)
                key = (
                    data.get("api_key")
                    or data.get("ANTHROPIC_API_KEY")
                    or data.get("anthropic_api_key")
                )
            except json.JSONDecodeError:
                key = raw

            if key:
                return key.strip()

        except SECRETS_CLIENT.exceptions.ResourceNotFoundException:
            continue
        except Exception as e:
            print(
                f"[byok] Failed to load Anthropic API key for tenant='{tenant_id}' "
                f"(secret='{secret_name}'): {e}",
                flush=True,
            )

    return None


def get_anthropic_key_for_tenant(tenant_id: str, byok_required: bool) -> str:
    """
    Strategy for Anthropic API key - Mixed:

    - Empty or 'default' tenant:
    - always uses DEFAULT_ANTHROPIC_API_KEY.
    - Managed tenant (byok_required=False):
    - tries BYOK (Secrets Manager);
    - if not found, falls back to the global key.
    - Byok_required tenant (byok_required=True):
    - tries BYOK;
    - if not found, ERROR (no fallback to the global key).
    """
    if not tenant_id or tenant_id == "default":
        return DEFAULT_ANTHROPIC_API_KEY

    tenant_key = _load_tenant_anthropic_key(tenant_id)
    if tenant_key:
        return tenant_key

    if byok_required:
        raise RuntimeError(
            f"BYOK required for tenant '{tenant_id}' but no Anthropic API key is configured"
        )

    return DEFAULT_ANTHROPIC_API_KEY


@lru_cache(maxsize=256)
def _load_tenant_deepseek_key(tenant_id: str) -> Optional[str]:
    """
    Attempt to load the tenant-specific DeepSeek API key (BYOK) in Secrets Manager.

    Checks both naming conventions:
    - {prefix}/{tenant_id}/deepseek (preferred)
    - {prefix}/{tenant_id}/deepseek_api_key (legacy)
    """
    if SECRETS_CLIENT is None:
        return None

    secret_names = [
        f"{TENANT_SECRET_PREFIX}/{tenant_id}/deepseek",
        f"{TENANT_SECRET_PREFIX}/{tenant_id}/deepseek_api_key",
    ]

    for secret_name in secret_names:
        try:
            resp = SECRETS_CLIENT.get_secret_value(SecretId=secret_name)
            raw = resp.get("SecretString") or ""
            if not raw:
                continue

            try:
                data = json.loads(raw)
                key = (
                    data.get("api_key")
                    or data.get("DEEPSEEK_API_KEY")
                    or data.get("deepseek_api_key")
                )
            except json.JSONDecodeError:
                key = raw

            if key:
                return key.strip()

        except SECRETS_CLIENT.exceptions.ResourceNotFoundException:
            continue
        except Exception as e:
            print(
                f"[byok] Failed to load DeepSeek API key for tenant='{tenant_id}' "
                f"(secret='{secret_name}'): {e}",
                flush=True,
            )

    return None


def get_deepseek_key_for_tenant(tenant_id: str, byok_required: bool) -> str:
    """
    Strategy for DeepSeek API key - Mixed:

    - Empty or 'default' tenant:
    - always uses DEFAULT_DEEPSEEK_API_KEY.
    - Managed tenant (byok_required=False):
    - tries BYOK (Secrets Manager);
    - if not found, falls back to the global key.
    - Byok_required tenant (byok_required=True):
    - tries BYOK;
    - if not found, ERROR (no fallback to the global key).
    """
    if not tenant_id or tenant_id == "default":
        return DEFAULT_DEEPSEEK_API_KEY

    tenant_key = _load_tenant_deepseek_key(tenant_id)
    if tenant_key:
        return tenant_key

    if byok_required:
        raise RuntimeError(
            f"BYOK required for tenant '{tenant_id}' but no DeepSeek API key is configured"
        )

    return DEFAULT_DEEPSEEK_API_KEY


@lru_cache(maxsize=256)
def _load_tenant_gemini_key(tenant_id: str) -> Optional[str]:
    """
    Attempt to load the tenant-specific Gemini API key (BYOK) in Secrets Manager.

    Checks both naming conventions:
    - {prefix}/{tenant_id}/gemini (preferred)
    - {prefix}/{tenant_id}/gemini_api_key (legacy)
    """
    if SECRETS_CLIENT is None:
        return None

    secret_names = [
        f"{TENANT_SECRET_PREFIX}/{tenant_id}/gemini",
        f"{TENANT_SECRET_PREFIX}/{tenant_id}/gemini_api_key",
    ]

    for secret_name in secret_names:
        try:
            resp = SECRETS_CLIENT.get_secret_value(SecretId=secret_name)
            raw = resp.get("SecretString") or ""
            if not raw:
                continue

            try:
                data = json.loads(raw)
                key = (
                    data.get("api_key")
                    or data.get("GEMINI_API_KEY")
                    or data.get("gemini_api_key")
                )
            except json.JSONDecodeError:
                key = raw

            if key:
                return key.strip()

        except SECRETS_CLIENT.exceptions.ResourceNotFoundException:
            continue
        except Exception as e:
            print(
                f"[byok] Failed to load Gemini API key for tenant='{tenant_id}' "
                f"(secret='{secret_name}'): {e}",
                flush=True,
            )

    return None


def get_gemini_key_for_tenant(tenant_id: str, byok_required: bool) -> str:
    """
    Strategy for Gemini API key - Mixed:

    - Empty or 'default' tenant:
    - always uses DEFAULT_GEMINI_API_KEY.
    - Managed tenant (byok_required=False):
    - tries BYOK (Secrets Manager);
    - if not found, falls back to the global key.
    - Byok_required tenant (byok_required=True):
    - tries BYOK;
    - if not found, ERROR (no fallback to the global key).
    """
    if not tenant_id or tenant_id == "default":
        return DEFAULT_GEMINI_API_KEY

    tenant_key = _load_tenant_gemini_key(tenant_id)
    if tenant_key:
        return tenant_key

    if byok_required:
        raise RuntimeError(
            f"BYOK required for tenant '{tenant_id}' but no Gemini API key is configured"
        )

    return DEFAULT_GEMINI_API_KEY


@lru_cache(maxsize=256)
def _load_tenant_mistral_key(tenant_id: str) -> Optional[str]:
    """
    Attempt to load the tenant-specific Mistral API key (BYOK) in Secrets Manager.
    No fallback applies here; only:
    - returns the key if found,
    - returns None if it doesn't exist/error.

    Checks both naming conventions:
    - {prefix}/{tenant_id}/mistral (preferred)
    - {prefix}/{tenant_id}/mistral_api_key (legacy)
    """
    if SECRETS_CLIENT is None:
        return None

    # Try both naming conventions
    secret_names = [
        f"{TENANT_SECRET_PREFIX}/{tenant_id}/mistral",
        f"{TENANT_SECRET_PREFIX}/{tenant_id}/mistral_api_key",
    ]

    for secret_name in secret_names:
        try:
            resp = SECRETS_CLIENT.get_secret_value(SecretId=secret_name)
            raw = resp.get("SecretString") or ""
            if not raw:
                continue

            try:
                data = json.loads(raw)
                key = (
                    data.get("api_key")
                    or data.get("MISTRAL_API_KEY")
                    or data.get("mistral_api_key")
                )
            except json.JSONDecodeError:
                key = raw

            if key:
                return key.strip()

        except SECRETS_CLIENT.exceptions.ResourceNotFoundException:
            continue
        except Exception as e:
            print(
                f"[byok] Failed to load Mistral API key for tenant='{tenant_id}' "
                f"(secret='{secret_name}'): {e}",
                flush=True,
            )

    return None


def get_mistral_key_for_tenant(tenant_id: str, byok_required: bool) -> str:
    """
    Strategy for Mistral API key - Mixed:

    - Empty or 'default' tenant:
    - always uses DEFAULT_MISTRAL_API_KEY.
    - Managed tenant (byok_required=False):
    - tries BYOK (Secrets Manager);
    - if not found, falls back to the global key.
    - Byok_required tenant (byok_required=True):
    - tries BYOK;
    - if not found, ERROR (no fallback to the global key).
    """
    if not tenant_id or tenant_id == "default":
        return DEFAULT_MISTRAL_API_KEY

    tenant_key = _load_tenant_mistral_key(tenant_id)
    if tenant_key:
        return tenant_key

    if byok_required:
        raise RuntimeError(
            f"BYOK required for tenant '{tenant_id}' but no Mistral API key is configured"
        )

    return DEFAULT_MISTRAL_API_KEY


@lru_cache(maxsize=256)
def _load_tenant_perplexity_key(tenant_id: str) -> Optional[str]:
    """
    Attempt to load the tenant-specific Perplexity API key (BYOK) in Secrets Manager.

    Checks both naming conventions:
    - {prefix}/{tenant_id}/perplexity (preferred)
    - {prefix}/{tenant_id}/perplexity_api_key (legacy)
    """
    if SECRETS_CLIENT is None:
        return None

    secret_names = [
        f"{TENANT_SECRET_PREFIX}/{tenant_id}/perplexity",
        f"{TENANT_SECRET_PREFIX}/{tenant_id}/perplexity_api_key",
    ]

    for secret_name in secret_names:
        try:
            resp = SECRETS_CLIENT.get_secret_value(SecretId=secret_name)
            raw = resp.get("SecretString") or ""
            if not raw:
                continue

            try:
                data = json.loads(raw)
                key = (
                    data.get("api_key")
                    or data.get("PERPLEXITY_API_KEY")
                    or data.get("perplexity_api_key")
                )
            except json.JSONDecodeError:
                key = raw

            if key:
                return key.strip()

        except SECRETS_CLIENT.exceptions.ResourceNotFoundException:
            continue
        except Exception as e:
            print(
                f"[byok] Failed to load Perplexity API key for tenant='{tenant_id}' "
                f"(secret='{secret_name}'): {e}",
                flush=True,
            )

    return None


def get_perplexity_key_for_tenant(tenant_id: str, byok_required: bool) -> str:
    """
    Strategy for Perplexity API key - Mixed:

    - Empty or 'default' tenant:
    - always uses DEFAULT_PERPLEXITY_API_KEY.
    - Managed tenant (byok_required=False):
    - tries BYOK (Secrets Manager);
    - if not found, falls back to the global key.
    - Byok_required tenant (byok_required=True):
    - tries BYOK;
    - if not found, ERROR (no fallback to the global key).
    """
    if not tenant_id or tenant_id == "default":
        return DEFAULT_PERPLEXITY_API_KEY

    tenant_key = _load_tenant_perplexity_key(tenant_id)
    if tenant_key:
        return tenant_key

    if byok_required:
        raise RuntimeError(
            f"BYOK required for tenant '{tenant_id}' but no Perplexity API key is configured"
        )

    return DEFAULT_PERPLEXITY_API_KEY


@lru_cache(maxsize=256)
def _load_tenant_groq_key(tenant_id: str) -> Optional[str]:
    """
    Attempt to load the tenant-specific Groq API key (BYOK) in Secrets Manager.

    Checks both naming conventions:
    - {prefix}/{tenant_id}/groq (preferred)
    - {prefix}/{tenant_id}/groq_api_key (legacy)
    """
    if SECRETS_CLIENT is None:
        return None

    secret_names = [
        f"{TENANT_SECRET_PREFIX}/{tenant_id}/groq",
        f"{TENANT_SECRET_PREFIX}/{tenant_id}/groq_api_key",
    ]

    for secret_name in secret_names:
        try:
            resp = SECRETS_CLIENT.get_secret_value(SecretId=secret_name)
            raw = resp.get("SecretString") or ""
            if not raw:
                continue

            try:
                data = json.loads(raw)
                key = (
                    data.get("api_key")
                    or data.get("GROQ_API_KEY")
                    or data.get("groq_api_key")
                )
            except json.JSONDecodeError:
                key = raw

            if key:
                return key.strip()

        except SECRETS_CLIENT.exceptions.ResourceNotFoundException:
            continue
        except Exception as e:
            print(
                f"[byok] Failed to load Groq API key for tenant='{tenant_id}' "
                f"(secret='{secret_name}'): {e}",
                flush=True,
            )

    return None


def get_groq_key_for_tenant(tenant_id: str, byok_required: bool) -> str:
    """
    Strategy for Groq API key - Mixed:

    - Empty or 'default' tenant:
    - always uses DEFAULT_GROQ_API_KEY.
    - Managed tenant (byok_required=False):
    - tries BYOK (Secrets Manager);
    - if not found, falls back to the global key.
    - Byok_required tenant (byok_required=True):
    - tries BYOK;
    - if not found, ERROR (no fallback to the global key).
    """
    if not tenant_id or tenant_id == "default":
        return DEFAULT_GROQ_API_KEY

    tenant_key = _load_tenant_groq_key(tenant_id)
    if tenant_key:
        return tenant_key

    if byok_required:
        raise RuntimeError(
            f"BYOK required for tenant '{tenant_id}' but no Groq API key is configured"
        )

    return DEFAULT_GROQ_API_KEY


@lru_cache(maxsize=256)
def _load_tenant_cerebras_key(tenant_id: str) -> Optional[str]:
    """
    Attempt to load the tenant-specific Cerebras API key (BYOK) in Secrets Manager.

    Checks both naming conventions:
    - {prefix}/{tenant_id}/cerebras (preferred)
    - {prefix}/{tenant_id}/cerebras_api_key (legacy)
    """
    if SECRETS_CLIENT is None:
        return None

    secret_names = [
        f"{TENANT_SECRET_PREFIX}/{tenant_id}/cerebras",
        f"{TENANT_SECRET_PREFIX}/{tenant_id}/cerebras_api_key",
    ]

    for secret_name in secret_names:
        try:
            resp = SECRETS_CLIENT.get_secret_value(SecretId=secret_name)
            raw = resp.get("SecretString") or ""
            if not raw:
                continue

            try:
                data = json.loads(raw)
                key = (
                    data.get("api_key")
                    or data.get("CEREBRAS_API_KEY")
                    or data.get("cerebras_api_key")
                )
            except json.JSONDecodeError:
                key = raw

            if key:
                return key.strip()

        except SECRETS_CLIENT.exceptions.ResourceNotFoundException:
            continue
        except Exception as e:
            print(
                f"[byok] Failed to load Cerebras API key for tenant='{tenant_id}' "
                f"(secret='{secret_name}'): {e}",
                flush=True,
            )

    return None


def get_cerebras_key_for_tenant(tenant_id: str, byok_required: bool) -> str:
    """
    Strategy for Cerebras API key - Mixed:

    - Empty or 'default' tenant:
    - always uses DEFAULT_CEREBRAS_API_KEY.
    - Managed tenant (byok_required=False):
    - tries BYOK (Secrets Manager);
    - if not found, falls back to the global key.
    - Byok_required tenant (byok_required=True):
    - tries BYOK;
    - if not found, ERROR (no fallback to the global key).
    """
    if not tenant_id or tenant_id == "default":
        return DEFAULT_CEREBRAS_API_KEY

    tenant_key = _load_tenant_cerebras_key(tenant_id)
    if tenant_key:
        return tenant_key

    if byok_required:
        raise RuntimeError(
            f"BYOK required for tenant '{tenant_id}' but no Cerebras API key is configured"
        )

    return DEFAULT_CEREBRAS_API_KEY


@lru_cache(maxsize=256)
def _load_tenant_cohere_key(tenant_id: str) -> Optional[str]:
    """
    Attempt to load the tenant-specific Cohere API key (BYOK) in Secrets Manager.

    Checks both naming conventions:
    - {prefix}/{tenant_id}/cohere (preferred)
    - {prefix}/{tenant_id}/cohere_api_key (legacy)
    """
    if SECRETS_CLIENT is None:
        return None

    secret_names = [
        f"{TENANT_SECRET_PREFIX}/{tenant_id}/cohere",
        f"{TENANT_SECRET_PREFIX}/{tenant_id}/cohere_api_key",
    ]

    for secret_name in secret_names:
        try:
            resp = SECRETS_CLIENT.get_secret_value(SecretId=secret_name)
            raw = resp.get("SecretString") or ""
            if not raw:
                continue

            try:
                data = json.loads(raw)
                key = (
                    data.get("api_key")
                    or data.get("COHERE_API_KEY")
                    or data.get("cohere_api_key")
                )
            except json.JSONDecodeError:
                key = raw

            if key:
                return key.strip()

        except SECRETS_CLIENT.exceptions.ResourceNotFoundException:
            continue
        except Exception as e:
            print(
                f"[byok] Failed to load Cohere API key for tenant='{tenant_id}' "
                f"(secret='{secret_name}'): {e}",
                flush=True,
            )

    return None


def get_cohere_key_for_tenant(tenant_id: str, byok_required: bool) -> str:
    """
    Strategy for Cohere API key - Mixed:

    - Empty or 'default' tenant:
    - always uses DEFAULT_COHERE_API_KEY.
    - Managed tenant (byok_required=False):
    - tries BYOK (Secrets Manager);
    - if not found, falls back to the global key.
    - Byok_required tenant (byok_required=True):
    - tries BYOK;
    - if not found, ERROR (no fallback to the global key).
    """
    if not tenant_id or tenant_id == "default":
        return DEFAULT_COHERE_API_KEY

    tenant_key = _load_tenant_cohere_key(tenant_id)
    if tenant_key:
        return tenant_key

    if byok_required:
        raise RuntimeError(
            f"BYOK required for tenant '{tenant_id}' but no Cohere API key is configured"
        )

    return DEFAULT_COHERE_API_KEY


@lru_cache(maxsize=256)
def _load_tenant_sambanova_key(tenant_id: str) -> Optional[str]:
    """
    Attempt to load the tenant-specific SambaNova API key (BYOK) in Secrets Manager.

    Checks both naming conventions:
    - {prefix}/{tenant_id}/sambanova (preferred)
    - {prefix}/{tenant_id}/sambanova_api_key (legacy)
    """
    if SECRETS_CLIENT is None:
        return None

    secret_names = [
        f"{TENANT_SECRET_PREFIX}/{tenant_id}/sambanova",
        f"{TENANT_SECRET_PREFIX}/{tenant_id}/sambanova_api_key",
    ]

    for secret_name in secret_names:
        try:
            resp = SECRETS_CLIENT.get_secret_value(SecretId=secret_name)
            raw = resp.get("SecretString") or ""
            if not raw:
                continue

            try:
                data = json.loads(raw)
                key = (
                    data.get("api_key")
                    or data.get("SAMBANOVA_API_KEY")
                    or data.get("sambanova_api_key")
                )
            except json.JSONDecodeError:
                key = raw

            if key:
                return key.strip()

        except SECRETS_CLIENT.exceptions.ResourceNotFoundException:
            continue
        except Exception as e:
            print(
                f"[byok] Failed to load SambaNova API key for tenant='{tenant_id}' "
                f"(secret='{secret_name}'): {e}",
                flush=True,
            )

    return None


def get_sambanova_key_for_tenant(tenant_id: str, byok_required: bool) -> str:
    """
    Strategy for SambaNova API key - Mixed:

    - Empty or 'default' tenant:
    - always uses DEFAULT_SAMBANOVA_API_KEY.
    - Managed tenant (byok_required=False):
    - tries BYOK (Secrets Manager);
    - if not found, falls back to the global key.
    - Byok_required tenant (byok_required=True):
    - tries BYOK;
    - if not found, ERROR (no fallback to the global key).
    """
    if not tenant_id or tenant_id == "default":
        return DEFAULT_SAMBANOVA_API_KEY

    tenant_key = _load_tenant_sambanova_key(tenant_id)
    if tenant_key:
        return tenant_key

    if byok_required:
        raise RuntimeError(
            f"BYOK required for tenant '{tenant_id}' but no SambaNova API key is configured"
        )

    return DEFAULT_SAMBANOVA_API_KEY