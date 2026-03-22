"""
MCP Server implementation for Lunar Router.

This server exposes Lunar Router functionality as MCP tools
for use with Claude Code, Claw, and other MCP-compatible clients.

Usage:
    # Run the server
    python -m lunar_router.mcp

    # Or use the CLI
    lunar-router mcp

Configuration in Claude Code (~/.claude/claude_code_config.json):
    {
      "mcpServers": {
        "lunar-router": {
          "command": "python",
          "args": ["-m", "lunar_router.mcp"]
        }
      }
    }
"""

import asyncio
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def create_server():
    """Create and configure the MCP server."""
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import Tool, TextContent
    except ImportError:
        raise ImportError(
            "MCP package required. Install with: pip install mcp"
        )

    server = Server("lunar-router")

    # Lazy loading of router components
    _router = None
    _clients = {}

    def get_router():
        nonlocal _router
        if _router is None:
            from ..loader import load_router
            try:
                _router = load_router(verbose=False)
            except FileNotFoundError:
                return None
        return _router

    def get_client(provider: str, model: str):
        key = f"{provider}:{model}"
        if key not in _clients:
            from ..models.llm_client import create_client
            _clients[key] = create_client(provider, model)
        return _clients[key]

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available tools."""
        return [
            Tool(
                name="lunar_route",
                description="Route a prompt to the best LLM based on semantic understanding. Returns the recommended model and expected performance.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The prompt to route"
                        },
                        "cost_weight": {
                            "type": "number",
                            "description": "Cost vs quality trade-off (0.0 = best quality, 1.0 = lowest cost)",
                            "default": 0.3
                        }
                    },
                    "required": ["prompt"]
                }
            ),
            Tool(
                name="lunar_generate",
                description="Generate a response using a specific LLM provider and model.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The prompt to send to the model"
                        },
                        "provider": {
                            "type": "string",
                            "description": "LLM provider (openai, anthropic, google, groq, mistral)",
                            "enum": ["openai", "anthropic", "google", "groq", "mistral"]
                        },
                        "model": {
                            "type": "string",
                            "description": "Model name (e.g., gpt-4o-mini, claude-3-5-haiku-20241022)"
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Maximum tokens to generate",
                            "default": 1000
                        },
                        "temperature": {
                            "type": "number",
                            "description": "Sampling temperature",
                            "default": 0.7
                        }
                    },
                    "required": ["prompt", "provider", "model"]
                }
            ),
            Tool(
                name="lunar_smart_generate",
                description="Automatically route and generate: selects the best model for the prompt and generates a response.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The prompt to process"
                        },
                        "cost_weight": {
                            "type": "number",
                            "description": "Cost vs quality trade-off (0.0 = best quality, 1.0 = lowest cost)",
                            "default": 0.3
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Maximum tokens to generate",
                            "default": 1000
                        }
                    },
                    "required": ["prompt"]
                }
            ),
            Tool(
                name="lunar_list_models",
                description="List all available models and their costs per provider.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "provider": {
                            "type": "string",
                            "description": "Filter by provider (optional)",
                            "enum": ["openai", "anthropic", "google", "groq", "mistral"]
                        }
                    }
                }
            ),
            Tool(
                name="lunar_compare",
                description="Compare responses from multiple models for the same prompt.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The prompt to send to all models"
                        },
                        "models": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "provider": {"type": "string"},
                                    "model": {"type": "string"}
                                }
                            },
                            "description": "List of provider/model pairs to compare"
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Maximum tokens per response",
                            "default": 500
                        }
                    },
                    "required": ["prompt", "models"]
                }
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Handle tool calls."""
        try:
            if name == "lunar_route":
                return await handle_route(arguments)
            elif name == "lunar_generate":
                return await handle_generate(arguments)
            elif name == "lunar_smart_generate":
                return await handle_smart_generate(arguments)
            elif name == "lunar_list_models":
                return await handle_list_models(arguments)
            elif name == "lunar_compare":
                return await handle_compare(arguments)
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]
        except Exception as e:
            logger.exception(f"Error in tool {name}")
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    async def handle_route(args: dict) -> list[TextContent]:
        """Handle lunar_route tool."""
        router = get_router()
        if router is None:
            return [TextContent(
                type="text",
                text="Router not available. Download weights first:\n  lunar-router download weights-mmlu-v1"
            )]

        prompt = args["prompt"]
        cost_weight = args.get("cost_weight", 0.3)

        router.cost_weight = cost_weight
        decision = router.route(prompt)

        result = {
            "selected_model": decision.selected_model,
            "expected_error": round(decision.expected_error, 4),
            "cost_adjusted_score": round(decision.cost_adjusted_score, 4),
            "cluster_id": decision.cluster_id,
            "top_models": dict(sorted(
                decision.all_scores.items(),
                key=lambda x: x[1]
            )[:5])
        }

        return [TextContent(
            type="text",
            text=f"Routing result:\n{json.dumps(result, indent=2)}"
        )]

    async def handle_generate(args: dict) -> list[TextContent]:
        """Handle lunar_generate tool."""
        provider = args["provider"]
        model = args["model"]
        prompt = args["prompt"]
        max_tokens = args.get("max_tokens", 1000)
        temperature = args.get("temperature", 0.7)

        client = get_client(provider, model)
        response = client.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

        result = {
            "model": f"{provider}/{model}",
            "response": response.text,
            "tokens": response.tokens_used,
            "latency_ms": round(response.latency_ms, 2)
        }

        return [TextContent(
            type="text",
            text=f"Response from {provider}/{model}:\n\n{response.text}\n\n---\nTokens: {response.tokens_used}, Latency: {response.latency_ms:.0f}ms"
        )]

    async def handle_smart_generate(args: dict) -> list[TextContent]:
        """Handle lunar_smart_generate tool."""
        router = get_router()
        if router is None:
            # Fallback to gpt-4o-mini if router not available
            args["provider"] = "openai"
            args["model"] = "gpt-4o-mini"
            return await handle_generate(args)

        prompt = args["prompt"]
        cost_weight = args.get("cost_weight", 0.3)
        max_tokens = args.get("max_tokens", 1000)

        # Route
        router.cost_weight = cost_weight
        decision = router.route(prompt)
        selected = decision.selected_model

        # Map model to provider
        model_to_provider = {
            "gpt-4o": "openai",
            "gpt-4o-mini": "openai",
            "gpt-4-turbo": "openai",
            "gpt-3.5-turbo": "openai",
            "claude-3-5-sonnet-20241022": "anthropic",
            "claude-3-5-haiku-20241022": "anthropic",
            "gemini-1.5-flash": "google",
            "gemini-1.5-pro": "google",
            "llama-3.1-8b-instant": "groq",
            "llama-3.3-70b-versatile": "groq",
            "mistral-large-latest": "mistral",
            "mistral-small-latest": "mistral",
        }

        provider = model_to_provider.get(selected, "openai")

        # Generate
        client = get_client(provider, selected)
        response = client.generate(prompt, max_tokens=max_tokens)

        return [TextContent(
            type="text",
            text=f"Routed to: {selected} (expected error: {decision.expected_error:.2%})\n\n{response.text}\n\n---\nTokens: {response.tokens_used}, Latency: {response.latency_ms:.0f}ms"
        )]

    async def handle_list_models(args: dict) -> list[TextContent]:
        """Handle lunar_list_models tool."""
        from ..models.llm_client import (
            OpenAIClient, AnthropicClient, GoogleClient,
            GroqClient, MistralClient
        )

        providers = {
            "openai": OpenAIClient.COSTS,
            "anthropic": AnthropicClient.COSTS,
            "google": GoogleClient.COSTS,
            "groq": GroqClient.COSTS,
            "mistral": MistralClient.COSTS,
        }

        filter_provider = args.get("provider")

        lines = ["Available Models:\n"]
        for provider_name, models in providers.items():
            if filter_provider and provider_name != filter_provider:
                continue

            lines.append(f"\n{provider_name.upper()}:")
            for model, cost in models.items():
                cost_str = f"${cost*1000:.4f}/1M tokens" if cost > 0 else "Free"
                lines.append(f"  - {model}: {cost_str}")

        return [TextContent(type="text", text="\n".join(lines))]

    async def handle_compare(args: dict) -> list[TextContent]:
        """Handle lunar_compare tool."""
        prompt = args["prompt"]
        models = args["models"]
        max_tokens = args.get("max_tokens", 500)

        results = []
        for spec in models:
            provider = spec["provider"]
            model = spec["model"]
            try:
                client = get_client(provider, model)
                response = client.generate(prompt, max_tokens=max_tokens)
                results.append({
                    "provider": provider,
                    "model": model,
                    "response": response.text,
                    "tokens": response.tokens_used,
                    "latency_ms": round(response.latency_ms, 2)
                })
            except Exception as e:
                results.append({
                    "provider": provider,
                    "model": model,
                    "error": str(e)
                })

        lines = [f"Comparison for: {prompt[:50]}...\n"]
        for r in results:
            lines.append(f"\n{'='*50}")
            lines.append(f"{r['provider']}/{r['model']}:")
            if "error" in r:
                lines.append(f"  Error: {r['error']}")
            else:
                lines.append(f"  Response: {r['response'][:200]}...")
                lines.append(f"  Tokens: {r['tokens']}, Latency: {r['latency_ms']}ms")

        return [TextContent(type="text", text="\n".join(lines))]

    return server


async def run_server():
    """Run the MCP server."""
    try:
        from mcp.server.stdio import stdio_server
    except ImportError:
        print("MCP package required. Install with: pip install mcp")
        return

    server = create_server()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


def main():
    """Entry point for the MCP server."""
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
