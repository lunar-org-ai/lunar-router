"""Outbound MCP client (P3.4).

OpenTracy has ALWAYS exposed an MCP server (``harness/introspection``)
so Claude Code can introspect itself. P3.4 adds the other direction:
the runtime CONNECTS to MCP servers configured by the operator, lists
their tools, and feeds those tools to the LLM during /run.

Public API:
  - ``runtime.mcp.client.list_tools_for_agent(agent_id)``  → cached catalog
  - ``runtime.mcp.client.call_tool(agent_id, name, args)`` → invoke

The async MCP SDK ships with ``mcp.client.stdio``. We wrap it in a
sync-friendly facade because the runtime's generate stage runs in a
threadpool (FastAPI sync handlers).
"""
