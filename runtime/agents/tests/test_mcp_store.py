"""Tests for the per-agent MCP server store (P3.4 T1)."""

from __future__ import annotations

import pytest

from runtime.agents.mcp import (
    MCPServer,
    add_server,
    enabled_servers,
    load,
    remove_server,
    save,
    update_server,
)


def test_load_missing_returns_empty(tmp_path):
    assert load("ghost", root=tmp_path) == []


def test_save_and_load_round_trip(tmp_path):
    servers = [
        MCPServer(name="fs", command="npx", args=["-y", "@mcp/fs", "/tmp"]),
        MCPServer(name="db", command="python", args=["-m", "mcp_db"], enabled=False),
    ]
    save("a", servers, root=tmp_path)
    loaded = load("a", root=tmp_path)
    assert len(loaded) == 2
    assert loaded[0].name == "fs"
    assert loaded[0].args == ["-y", "@mcp/fs", "/tmp"]
    assert loaded[1].enabled is False


def test_add_server_appends(tmp_path):
    add_server("a", MCPServer(name="fs", command="npx"), root=tmp_path)
    add_server("a", MCPServer(name="db", command="python"), root=tmp_path)
    out = load("a", root=tmp_path)
    assert [s.name for s in out] == ["fs", "db"]


def test_add_server_rejects_duplicate(tmp_path):
    add_server("a", MCPServer(name="fs", command="npx"), root=tmp_path)
    with pytest.raises(ValueError):
        add_server("a", MCPServer(name="fs", command="other"), root=tmp_path)


def test_update_server_partial(tmp_path):
    add_server("a", MCPServer(name="fs", command="npx", enabled=True), root=tmp_path)
    update_server("a", "fs", enabled=False, description="paused", root=tmp_path)
    out = load("a", root=tmp_path)
    assert out[0].enabled is False
    assert out[0].description == "paused"
    assert out[0].command == "npx"  # untouched


def test_update_server_missing_raises(tmp_path):
    with pytest.raises(KeyError):
        update_server("a", "ghost", enabled=False, root=tmp_path)


def test_remove_server(tmp_path):
    add_server("a", MCPServer(name="fs", command="npx"), root=tmp_path)
    add_server("a", MCPServer(name="db", command="python"), root=tmp_path)
    remove_server("a", "fs", root=tmp_path)
    out = load("a", root=tmp_path)
    assert [s.name for s in out] == ["db"]


def test_remove_server_missing_is_noop(tmp_path):
    add_server("a", MCPServer(name="fs", command="npx"), root=tmp_path)
    remove_server("a", "ghost", root=tmp_path)
    out = load("a", root=tmp_path)
    assert len(out) == 1


def test_enabled_servers_filters(tmp_path):
    save("a", [
        MCPServer(name="fs", command="npx", enabled=True),
        MCPServer(name="db", command="python", enabled=False),
    ], root=tmp_path)
    enabled = enabled_servers("a", root=tmp_path)
    assert [s.name for s in enabled] == ["fs"]


def test_invalid_transport_normalized(tmp_path):
    save("a", [
        MCPServer(name="x", command="npx", transport="telepathy"),  # invalid
    ], root=tmp_path)
    # Reload — should be normalized to stdio on read
    out = load("a", root=tmp_path)
    assert out[0].transport == "stdio"
