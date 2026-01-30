import time
from typing import Tuple, Dict, Any, List, Optional
from litellm import acompletion, token_counter
from .adapters import ProviderAdapter, _extract_prompt
from .helpers.secrets_config import is_byok_required, get_bedrock_credentials_for_tenant
from .helpers.error_classifier import classify_error
from .models.error_types import ErrorCategory

class BedrockAdapter(ProviderAdapter):
    """
    Adapter for Bedrock models via LiteLLM with:
    - Byok per tenant (Secrets Manager)
    - Controlled fallback according to policy (managed vs byok_required)    
    - Streaming + metrics (ttft, latency, tokens)
    """

    def __init__(self, name: str, logical_model: str, model_name: str, region: str = "us-east-1", modelid: Optional[str] = None):
        super().__init__(name=name, model=logical_model)
        self.model_name = model_name
        self.modelid = modelid
        self.region = region

    def _ensure_messages(self, req: Dict[str, Any]) -> List[Dict[str, str]]:
        if "messages" in req and isinstance(req["messages"], list):
            return req["messages"]

        prompt = _extract_prompt(req)
        return [{"role": "user", "content": prompt}]

    def _count_tokens(self, messages, completion_text: str) -> Tuple[int, int]:
        # Use the original model name for token counting (without bedrock/ prefix)
        token_model = self.model_name.replace("bedrock/", "") if self.model_name.startswith("bedrock/") else self.model_name
        
        # tokens in
        try:
            ti = token_counter(model=token_model, messages=messages) or 0
        except Exception:
            ti = 0

        # tokens out
        try:
            to = token_counter(model=token_model, text=completion_text) or 0
        except Exception:
            to = max(5, int(len(completion_text.split()) * 1.3))

        if ti == 0:
            text = " ".join([m.get("content", "") for m in messages])
            ti = max(1, int(len(text.split()) * 1.2))

        return int(ti), int(to)



    async def send(self, req: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        tenant_id = str(req.get("tenant") or "default")
        stream = req.get("stream", False)
        byok_required = is_byok_required(tenant_id)

        try:
            print(f"Getting AWS credentials for Bedrock, tenant '{tenant_id}', BYOK required: {byok_required}", flush=True)
            aws_creds = get_bedrock_credentials_for_tenant(tenant_id, byok_required)
        except RuntimeError as e:
            return (
                {
                    "error": str(e),
                    "error_category": ErrorCategory.AUTH_ERROR.value,
                    "error_details": {
                        "exception_type": type(e).__name__,
                        "provider": self.name,
                    },
                },
                {
                    "ttft_ms": 0.0,
                    "latency_ms": 0.0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "error": 1.0,
                },
            )

        messages = self._ensure_messages(req)
        start = time.time()
        ttft_ms: Optional[float] = None
        text = ""

        try:
            # Use modelid if available, otherwise use model_name
            target_model = self.modelid if self.modelid else self.model_name

            # Remove region prefix (us., eu., ap-, etc.) from model name
            # AWS uses format like "us.meta.llama..." but LiteLLM expects "meta.llama..."
            if target_model and "." in target_model:
                parts = target_model.split(".", 1)
                # Check if first part is a region prefix (2-3 chars like us, eu, ap)
                if len(parts[0]) <= 3 and parts[0].lower() in ["us", "eu", "ap", "sa", "ca", "me", "af"]:
                    target_model = parts[1]
                    print(f"[BedrockAdapter] Removed region prefix, model: {target_model}", flush=True)

            # Ensure model name has bedrock/ prefix for LiteLLM
            litellm_model = target_model
            if not litellm_model.startswith("bedrock/"):
                litellm_model = f"bedrock/{target_model}"

            # Build kwargs for LiteLLM acompletion
            completion_kwargs = {
                "model": litellm_model,
                "messages": messages,
                "stream": stream,
                "aws_region_name": self.region,
            }

            if aws_creds:
                # Use tenant's BYOK IAM credentials
                completion_kwargs["aws_access_key_id"] = aws_creds["aws_access_key_id"]
                completion_kwargs["aws_secret_access_key"] = aws_creds["aws_secret_access_key"]
                completion_kwargs["aws_region_name"] = aws_creds.get("aws_region_name", self.region)
                print(f"[BedrockAdapter] Using BYOK IAM credentials for tenant '{tenant_id}', region: {completion_kwargs['aws_region_name']}", flush=True)
            else:
                # Use default credentials (IAM role / environment)
                print(f"[BedrockAdapter] Using default AWS credentials (IAM/environment), region: {self.region}", flush=True)

            print(f"[BedrockAdapter] Using model '{litellm_model}' (modelid: '{self.modelid}', model_name: '{self.model_name}')", flush=True)

            response = await acompletion(**completion_kwargs)

            if stream:
                ti = self._estimate_tokens_in(messages)
                
                return (
                    {
                        "stream": self._stream_with_metrics(response, start),
                        "messages": messages,
                        "model_name": self.model_name,
                    },
                    {
                        "ttft_ms": 0.0,  # Will be calculated during streaming
                        "latency_ms": 0.0,  # Will be calculated at the end
                        "tokens_in": ti,
                        "tokens_out": 0,
                        "error": 0.0,
                    },
                )
            else:
                # Handle non-streaming response (ModelResponse object)
                latency_ms = (time.time() - start) * 1000.0
                ttft_ms = latency_ms  # For non-streaming, ttft equals total latency
                
                # Extract text from ModelResponse
                try:
                    text = response.choices[0].message.content or ""
                except Exception:
                    text = ""

                # Use fast estimation instead of blocking token_counter
                ti = self._estimate_tokens_in(messages)
                to = self._estimate_tokens_out(text)

                return (
                    {"text": text},
                    {
                        "ttft_ms": float(ttft_ms),
                        "latency_ms": float(latency_ms),
                        "tokens_in": ti,
                        "tokens_out": to,
                        "error": 0.0,
                    },
                )

        except Exception as e:
            latency_ms = (time.time() - start) * 1000.0
            error_category = classify_error(e)
            return (
                {
                    "error": f"bedrock error: {e}",
                    "error_category": error_category,
                    "error_details": {
                        "exception_type": type(e).__name__,
                        "provider": self.name,
                    },
                },
                {
                    "ttft_ms": float(ttft_ms or 0.0),
                    "latency_ms": float(latency_ms),
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "error": 1.0,
                },
            )

    def healthy(self) -> Dict[str, Any]:
        """
        Generic Health for Bedrock:
        - Always returns ok=True because Bedrock uses AWS credentials
          (IAM role, environment variables, or BYOK) which are evaluated at runtime.
        - BYOK-specific checks are evaluated at runtime in send().
        """
        return {
            "ok": True,
            "err_rate": 0.0,
            "headroom": 1.0,
        }