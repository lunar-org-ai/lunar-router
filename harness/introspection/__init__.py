"""harness.introspection — tools for asking the harness about itself.

Two consumers, one source:
  - Claude Code (dev): registered via .mcp.json, talks via stdio.
  - Anthropic SDK + MCP client (prod): same server, same tools, embedded
    in the runtime serving the UI's Introspection tab.
"""
