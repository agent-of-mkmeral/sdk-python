"""Tests for MCPRegistry and load_mcp_servers."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strands.mcp.registry import MCPRegistry, load_mcp_servers, _build_client_from_config
from strands.tools.mcp.mcp_client import MCPClient


class TestMCPRegistry:
    """Tests for the MCPRegistry class."""

    def test_init_empty(self):
        reg = MCPRegistry()
        assert reg.server_names == []
        assert reg.clients == {}

    def test_init_with_clients(self):
        c1 = MagicMock(spec=MCPClient)
        c2 = MagicMock(spec=MCPClient)
        reg = MCPRegistry({"a": c1, "b": c2})
        assert set(reg.server_names) == {"a", "b"}

    def test_add_client(self):
        reg = MCPRegistry()
        c = MagicMock(spec=MCPClient)
        reg.add_client("x", c)
        assert "x" in reg.server_names
        assert reg.clients["x"] is c

    def test_clients_returns_copy(self):
        c = MagicMock(spec=MCPClient)
        reg = MCPRegistry({"a": c})
        d = reg.clients
        d["b"] = MagicMock()
        # Original should not be modified
        assert "b" not in reg.server_names

    def test_register_tools_requires_tool_registry(self):
        reg = MCPRegistry({"a": MagicMock(spec=MCPClient)})
        with pytest.raises(TypeError, match="Expected ToolRegistry"):
            reg.register_tools("not-a-registry", MagicMock())

    def test_register_tools_calls_process_tools(self):
        from strands.tools.registry import ToolRegistry

        client = MagicMock(spec=MCPClient)
        client._logging_callback = None

        tool_reg = MagicMock(spec=ToolRegistry)
        hook_reg = MagicMock()

        reg = MCPRegistry({"srv": client})
        reg.register_tools(tool_reg, hook_reg)

        tool_reg.process_tools.assert_called_once()
        # The argument should be a list containing the client
        args = tool_reg.process_tools.call_args[0][0]
        assert client in args


class TestLoadMcpServers:
    """Tests for load_mcp_servers factory function."""

    def test_load_from_json_string(self):
        cfg = json.dumps({
            "mcpServers": {
                "fs": {"command": "cat", "args": ["--help"]}
            }
        })
        with patch("strands.mcp.registry._build_client_from_config") as mock_build:
            mock_build.return_value = MagicMock(spec=MCPClient)
            result = load_mcp_servers(cfg)
        assert "fs" in result
        mock_build.assert_called_once_with("fs", {"command": "cat", "args": ["--help"]})

    def test_load_skips_disabled(self):
        cfg = {
            "mcpServers": {
                "disabled_one": {"command": "echo", "disabled": True},
                "enabled_one": {"command": "echo"},
            }
        }
        with patch("strands.mcp.registry._build_client_from_config") as mock_build:
            mock_build.return_value = MagicMock(spec=MCPClient)
            result = load_mcp_servers(cfg)
        assert "disabled_one" not in result
        assert "enabled_one" in result

    def test_load_inner_dict_format(self):
        """Accept the inner dict directly (no mcpServers wrapper)."""
        cfg = {"srv": {"command": "echo"}}
        with patch("strands.mcp.registry._build_client_from_config") as mock_build:
            mock_build.return_value = MagicMock(spec=MCPClient)
            result = load_mcp_servers(cfg)
        assert "srv" in result

    def test_load_handles_build_error(self):
        """If a server config is bad, skip it and continue."""
        cfg = {
            "mcpServers": {
                "bad": {"no_command_no_url": True},
                "good": {"command": "echo"},
            }
        }
        with patch("strands.mcp.registry._build_client_from_config") as mock_build:
            def side_effect(name, cfg):
                if name == "bad":
                    raise ValueError("test error")
                return MagicMock(spec=MCPClient)
            mock_build.side_effect = side_effect
            result = load_mcp_servers(cfg)
        assert "bad" not in result
        assert "good" in result


class TestBuildClientFromConfig:
    """Tests for _build_client_from_config."""

    def test_stdio_transport(self):
        cfg = {"command": "echo", "args": ["hello"], "env": {"FOO": "bar"}}
        client = _build_client_from_config("test", cfg)
        assert isinstance(client, MCPClient)
        assert client._prefix == "test"

    def test_url_transport_sse(self):
        cfg = {"url": "https://example.com/sse"}
        client = _build_client_from_config("test", cfg)
        assert isinstance(client, MCPClient)

    def test_url_transport_http(self):
        cfg = {"url": "https://example.com/mcp"}
        client = _build_client_from_config("test", cfg)
        assert isinstance(client, MCPClient)

    def test_missing_command_and_url_raises(self):
        with pytest.raises(ValueError, match="must specify either 'command' or 'url'"):
            _build_client_from_config("test", {"some_key": "value"})

    def test_custom_prefix(self):
        cfg = {"command": "echo", "prefix": "my_prefix"}
        client = _build_client_from_config("test", cfg)
        assert client._prefix == "my_prefix"

    def test_disabled_tools_creates_filter(self):
        cfg = {"command": "echo", "disabledTools": ["tool_a", "tool_b"]}
        client = _build_client_from_config("test", cfg)
        assert client._tool_filters is not None
        assert "tool_a" in client._tool_filters["rejected"]
        assert "tool_b" in client._tool_filters["rejected"]
