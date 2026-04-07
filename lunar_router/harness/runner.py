"""AgentRunner — loads an agent .md, calls LLM, parses structured output.

Improvements over v1:
- JSON retry: asks LLM to fix malformed output
- Output schema validation: checks required fields
- Tool execution timeout (30s)
- Tool response truncation (prevents token overflow)
- Execution metrics (duration, tokens, model)
- Engine URL from env var
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Optional

import httpx

from .registry import AgentConfig, AgentRegistry
from .tools import ToolKit
from .memory_store import MemoryEntry, MemoryStore

logger = logging.getLogger(__name__)

ENGINE_URL = os.environ.get("LUNAR_ENGINE_URL", "http://localhost:8080")
TOOL_TIMEOUT = 30  # seconds
TOOL_RESULT_MAX_CHARS = 4000  # truncate tool results fed back to LLM


def _strip_markdown_json(text: str) -> str:
    """Extract JSON from LLM response, handling markdown code fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        start = 1
        end = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end = i
                break
        text = "\n".join(lines[start:end]).strip()
    return text


def _render_template(template: str, context: dict[str, Any]) -> str:
    """Simple {variable} replacement in prompt templates.

    Only replaces keys that exist in context — unknown placeholders are left as-is.
    """
    result = template
    for key, value in context.items():
        result = result.replace(f"{{{key}}}", str(value))
    return result


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... (truncated, {len(text)} total chars)"


class AgentResult:
    """Result of an agent execution with metadata."""

    def __init__(
        self,
        data: dict[str, Any],
        *,
        agent: str = "",
        model: str = "",
        duration_ms: float = 0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        tool_calls: int = 0,
        retried: bool = False,
    ):
        self.data = data
        self.agent = agent
        self.model = model
        self.duration_ms = duration_ms
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.tool_calls = tool_calls
        self.retried = retried

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.data,
            "_meta": {
                "agent": self.agent,
                "model": self.model,
                "duration_ms": round(self.duration_ms, 1),
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "tool_calls": self.tool_calls,
                "retried": self.retried,
            },
        }

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


class AgentRunner:
    """Loads an agent .md, sends prompt to LLM, parses structured output.

    All LLM calls use X-Lunar-Internal: true to prevent trace contamination.
    """

    def __init__(
        self,
        engine_url: Optional[str] = None,
        registry: Optional[AgentRegistry] = None,
        toolkit: Optional[ToolKit] = None,
        memory_store: Optional[MemoryStore] = None,
        record_memory: bool = False,
    ):
        self.engine_url = engine_url or ENGINE_URL
        self.registry = registry or AgentRegistry()
        self.toolkit = toolkit or ToolKit(engine_url=self.engine_url)
        self.memory_store = memory_store
        self.record_memory = record_memory

    def _record(self, result: AgentResult, user_input: str) -> None:
        """Persist an AgentResult to memory. Never raises."""
        if not (self.record_memory and self.memory_store):
            return
        try:
            entry = MemoryEntry.from_agent_result(result, user_input)
            self.memory_store.save(entry)
        except Exception as e:
            logger.debug(f"Memory recording failed: {e}")

    async def run(
        self,
        agent_name: str,
        user_input: str,
        context: Optional[dict[str, Any]] = None,
    ) -> AgentResult:
        """Execute an agent and return structured output.

        1. Load agent .md (system prompt + config)
        2. Render context variables into user_input
        3. Call LLM via Go engine
        4. Parse JSON response (with retry on failure)
        5. Validate against output_schema
        """
        start = time.time()

        config = self.registry.get(agent_name)
        if config is None:
            raise ValueError(f"Agent '{agent_name}' not found")

        if context:
            user_input = _render_template(user_input, context)

        response_text, usage = await self._call_llm(config, user_input)

        if config.output_schema.type == "json":
            data, retried = await self._parse_json_with_retry(response_text, config)
        else:
            data, retried = {"text": response_text}, False

        # Validate required fields
        self._validate_schema(data, config)

        result = AgentResult(
            data=data,
            agent=agent_name,
            model=config.model,
            duration_ms=(time.time() - start) * 1000,
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
            retried=retried,
        )
        self._record(result, user_input)
        return result

    async def run_with_tools(
        self,
        agent_name: str,
        user_input: str,
        max_turns: int = 5,
    ) -> AgentResult:
        """Execute an agent with tool access (multi-turn)."""
        start = time.time()
        tool_call_count = 0

        config = self.registry.get(agent_name)
        if config is None:
            raise ValueError(f"Agent '{agent_name}' not found")

        tools_desc = "\n".join(
            f"- {t['name']}: {t['description']}" for t in self.toolkit.available()
        )
        system = (
            f"{config.system_prompt}\n\n"
            f"You have access to these tools:\n{tools_desc}\n\n"
            f'To call a tool, respond with: {{"tool": "tool_name", "args": {{...}}}}\n'
            f"When you have your final answer, respond with the result JSON directly (no tool wrapper)."
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_input},
        ]

        total_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

        for turn in range(max_turns):
            response_text, usage = await self._call_llm_messages(config, messages)
            total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            total_usage["completion_tokens"] += usage.get("completion_tokens", 0)

            try:
                parsed = json.loads(_strip_markdown_json(response_text))
                if isinstance(parsed, dict) and "tool" in parsed:
                    tool_name = parsed["tool"]
                    tool_args = parsed.get("args", {})
                    tool_call_count += 1

                    tool_result = await self._execute_tool(tool_name, tool_args)

                    # Truncate large results
                    result_str = json.dumps(tool_result, default=str)
                    result_str = _truncate(result_str, TOOL_RESULT_MAX_CHARS)

                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({
                        "role": "user",
                        "content": f"Tool result for {tool_name}:\n{result_str}",
                    })
                    continue
                else:
                    result = AgentResult(
                        data=parsed if isinstance(parsed, dict) else {"value": parsed},
                        agent=agent_name,
                        model=config.model,
                        duration_ms=(time.time() - start) * 1000,
                        tokens_in=total_usage["prompt_tokens"],
                        tokens_out=total_usage["completion_tokens"],
                        tool_calls=tool_call_count,
                    )
                    self._record(result, user_input)
                    return result
            except json.JSONDecodeError:
                result = AgentResult(
                    data={"text": response_text},
                    agent=agent_name,
                    model=config.model,
                    duration_ms=(time.time() - start) * 1000,
                    tokens_in=total_usage["prompt_tokens"],
                    tokens_out=total_usage["completion_tokens"],
                    tool_calls=tool_call_count,
                )
                self._record(result, user_input)
                return result

        result = AgentResult(
            data={"error": "Max tool turns exhausted", "last_response": response_text},
            agent=agent_name,
            model=config.model,
            duration_ms=(time.time() - start) * 1000,
            tokens_in=total_usage["prompt_tokens"],
            tokens_out=total_usage["completion_tokens"],
            tool_calls=tool_call_count,
        )
        self._record(result, user_input)
        return result

    async def _execute_tool(self, tool_name: str, tool_args: dict) -> dict:
        """Execute a tool with timeout and error handling."""
        tool_fn = self.toolkit.get(tool_name)
        if tool_fn is None:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            result = await asyncio.wait_for(
                tool_fn(**tool_args),
                timeout=TOOL_TIMEOUT,
            )
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Tool {tool_name} timed out after {TOOL_TIMEOUT}s")
            return {"error": f"Tool {tool_name} timed out after {TOOL_TIMEOUT}s"}
        except Exception as e:
            logger.warning(f"Tool {tool_name} failed: {e}")
            return {"error": f"Tool {tool_name} failed: {str(e)}"}

    async def _call_llm(self, config: AgentConfig, user_input: str) -> tuple[str, dict]:
        """Single-turn LLM call. Returns (text, usage)."""
        messages = [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": user_input},
        ]
        return await self._call_llm_messages(config, messages)

    async def _call_llm_messages(self, config: AgentConfig, messages: list[dict]) -> tuple[str, dict]:
        """Call Go engine chat completions. Returns (text, usage)."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{self.engine_url}/v1/chat/completions",
                headers={"X-Lunar-Internal": "true"},
                json={
                    "model": config.model,
                    "messages": messages,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                },
            )
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return text, usage

    async def _parse_json_with_retry(
        self, text: str, config: AgentConfig,
    ) -> tuple[dict, bool]:
        """Parse JSON from LLM response. On failure, asks LLM to fix it."""
        cleaned = _strip_markdown_json(text)
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                parsed = parsed[0]
            elif not isinstance(parsed, dict):
                parsed = {"value": parsed}
            return parsed, False
        except json.JSONDecodeError as e:
            logger.debug(f"Agent {config.name}: JSON parse failed, retrying with LLM")

        # Retry: ask LLM to fix the JSON
        try:
            fix_prompt = (
                f"Your previous response was not valid JSON. Here is what you returned:\n\n"
                f"```\n{text[:2000]}\n```\n\n"
                f"Please return ONLY valid JSON with no extra text."
            )
            retry_text, _ = await self._call_llm(config, fix_prompt)
            cleaned = _strip_markdown_json(retry_text)
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                parsed = {"value": parsed}
            logger.info(f"Agent {config.name}: JSON fixed on retry")
            return parsed, True
        except Exception:
            logger.warning(f"Agent {config.name}: JSON parse failed after retry")
            return {"raw_text": text, "parse_error": str(e)}, True

    def _validate_schema(self, data: dict, config: AgentConfig) -> None:
        """Warn if required output_schema fields are missing."""
        if "parse_error" in data:
            return
        required = config.output_schema.fields
        for field_name, field_type in required.items():
            if field_name not in data:
                logger.warning(
                    f"Agent {config.name}: missing field '{field_name}' "
                    f"(expected {field_type}) in output"
                )
