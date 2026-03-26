"""Tests for Option B (MCP as Pass-Through) changes.

Tests cover:
1. Bug fix: _meta extraction from arguments and pass-through as meta= to call_tool()
2. Bug fix: isError field populated on MCPToolResult
3. Callback wiring: sampling, list_roots, logging, progress passed to ClientSession
4. Default logging callback routes to Python logging
5. Config loading: load_mcp_servers() from dict, JSON string, and file
6. MCP events: hook event hierarchy
"""

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from mcp.types import CallToolResult as MCPCallToolResult
from mcp.types import LoggingMessageNotificationParams, TextContent

from strands.hooks.mcp_events import (
    MCPLoggingEvent,
    MCPProgressEvent,
    MCPSamplingRequestEvent,
    MCPServerConnectedEvent,
    MCPServerDisconnectedEvent,
    MCPToolCallEndEvent,
    MCPToolCallStartEvent,
)
from strands.hooks.registry import HookRegistry
from strands.tools.mcp.mcp_client import MCPClient, _default_logging_callback
from strands.tools.mcp.mcp_config import _build_transport, _parse_config, load_mcp_servers
from strands.tools.mcp.mcp_types import MCPToolResult


# ==================================================================================
# Bug Fix Tests: _meta extraction
# ==================================================================================


class TestMetaExtraction:
    """Test that _meta is extracted from arguments and passed to session.call_tool()."""

    def test_meta_extracted_from_arguments(self, mock_transport, mock_session):
        """_meta should be popped from arguments and passed as meta= to call_tool."""
        mock_call_result = MCPCallToolResult(
            content=[TextContent(type="text", text="result")],
            isError=False,
        )
        mock_session.call_tool = AsyncMock(return_value=mock_call_result)

        client = MCPClient(mock_transport["transport_callable"])
        with client:
            result = client.call_tool_sync(
                "tool-use-1",
                "my_tool",
                {"arg1": "value1", "_meta": {"progressToken": "tok-123"}},
            )

        # Verify call_tool was called with meta extracted
        mock_session.call_tool.assert_called_once()
        call_args = mock_session.call_tool.call_args
        # Arguments should not contain _meta
        assert "_meta" not in call_args[0][1]  # positional arg[1] = arguments
        assert call_args[0][1] == {"arg1": "value1"}
        # meta should be passed as keyword
        assert call_args[1]["meta"] == {"progressToken": "tok-123"}

    def test_no_meta_in_arguments(self, mock_transport, mock_session):
        """When no _meta in arguments, meta= should be None."""
        mock_call_result = MCPCallToolResult(
            content=[TextContent(type="text", text="result")],
            isError=False,
        )
        mock_session.call_tool = AsyncMock(return_value=mock_call_result)

        client = MCPClient(mock_transport["transport_callable"])
        with client:
            result = client.call_tool_sync(
                "tool-use-2",
                "my_tool",
                {"arg1": "value1"},
            )

        mock_session.call_tool.assert_called_once()
        call_args = mock_session.call_tool.call_args
        assert call_args[1]["meta"] is None

    def test_meta_does_not_mutate_original_arguments(self, mock_transport, mock_session):
        """Extracting _meta should not mutate the caller's original dict."""
        mock_call_result = MCPCallToolResult(
            content=[TextContent(type="text", text="ok")],
            isError=False,
        )
        mock_session.call_tool = AsyncMock(return_value=mock_call_result)

        original_args = {"arg1": "value1", "_meta": {"progressToken": "tok-456"}}

        client = MCPClient(mock_transport["transport_callable"])
        with client:
            client.call_tool_sync("tool-use-3", "my_tool", original_args)

        # Original dict should still have _meta
        assert "_meta" in original_args

    def test_none_arguments_no_crash(self, mock_transport, mock_session):
        """When arguments is None, should not crash."""
        mock_call_result = MCPCallToolResult(
            content=[TextContent(type="text", text="ok")],
            isError=False,
        )
        mock_session.call_tool = AsyncMock(return_value=mock_call_result)

        client = MCPClient(mock_transport["transport_callable"])
        with client:
            result = client.call_tool_sync("tool-use-4", "my_tool", None)

        assert result["status"] == "success"


# ==================================================================================
# Bug Fix Tests: isError field on MCPToolResult
# ==================================================================================


class TestIsErrorField:
    """Test that isError is populated on MCPToolResult."""

    def test_is_error_true_when_tool_reports_error(self, mock_transport, mock_session):
        """isError should be True when MCP server reports error."""
        mock_call_result = MCPCallToolResult(
            content=[TextContent(type="text", text="something failed")],
            isError=True,
        )
        mock_session.call_tool = AsyncMock(return_value=mock_call_result)

        client = MCPClient(mock_transport["transport_callable"])
        with client:
            result = client.call_tool_sync("tool-use-err", "bad_tool", {})

        assert result["isError"] is True
        assert result["status"] == "error"

    def test_is_error_false_on_success(self, mock_transport, mock_session):
        """isError should be False on successful tool execution."""
        mock_call_result = MCPCallToolResult(
            content=[TextContent(type="text", text="great success")],
            isError=False,
        )
        mock_session.call_tool = AsyncMock(return_value=mock_call_result)

        client = MCPClient(mock_transport["transport_callable"])
        with client:
            result = client.call_tool_sync("tool-use-ok", "good_tool", {})

        assert result["isError"] is False
        assert result["status"] == "success"

    def test_is_error_true_on_execution_error(self, mock_transport, mock_session):
        """isError should be True when tool execution raises exception."""
        mock_session.call_tool = AsyncMock(side_effect=RuntimeError("boom"))

        client = MCPClient(mock_transport["transport_callable"])
        with client:
            result = client.call_tool_sync("tool-use-exc", "crash_tool", {})

        assert result["isError"] is True
        assert result["status"] == "error"


# ==================================================================================
# Callback Wiring Tests
# ==================================================================================


class TestCallbackWiring:
    """Test that all callbacks are wired through to ClientSession."""

    def test_sampling_callback_passed_to_session(self, mock_transport):
        """sampling_callback should be passed to ClientSession constructor."""
        mock_sampling = MagicMock()

        with patch("strands.tools.mcp.mcp_client.ClientSession") as MockSession:
            mock_session_instance = AsyncMock()
            mock_init_result = MagicMock()
            mock_init_result.instructions = None
            mock_session_instance.initialize = AsyncMock(return_value=mock_init_result)
            mock_session_instance.get_server_capabilities = MagicMock(return_value=None)

            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__.return_value = mock_session_instance
            MockSession.return_value = mock_session_cm

            client = MCPClient(
                mock_transport["transport_callable"],
                sampling_callback=mock_sampling,
            )
            with client:
                pass

            # Verify ClientSession was created with sampling_callback
            MockSession.assert_called_once()
            call_kwargs = MockSession.call_args[1]
            assert call_kwargs["sampling_callback"] is mock_sampling

    def test_list_roots_callback_passed_to_session(self, mock_transport):
        """list_roots_callback should be passed to ClientSession constructor."""
        mock_roots = MagicMock()

        with patch("strands.tools.mcp.mcp_client.ClientSession") as MockSession:
            mock_session_instance = AsyncMock()
            mock_init_result = MagicMock()
            mock_init_result.instructions = None
            mock_session_instance.initialize = AsyncMock(return_value=mock_init_result)
            mock_session_instance.get_server_capabilities = MagicMock(return_value=None)

            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__.return_value = mock_session_instance
            MockSession.return_value = mock_session_cm

            client = MCPClient(
                mock_transport["transport_callable"],
                list_roots_callback=mock_roots,
            )
            with client:
                pass

            call_kwargs = MockSession.call_args[1]
            assert call_kwargs["list_roots_callback"] is mock_roots

    def test_logging_callback_passed_to_session(self, mock_transport):
        """logging_callback should be passed to ClientSession constructor."""
        mock_logging = MagicMock()

        with patch("strands.tools.mcp.mcp_client.ClientSession") as MockSession:
            mock_session_instance = AsyncMock()
            mock_init_result = MagicMock()
            mock_init_result.instructions = None
            mock_session_instance.initialize = AsyncMock(return_value=mock_init_result)
            mock_session_instance.get_server_capabilities = MagicMock(return_value=None)

            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__.return_value = mock_session_instance
            MockSession.return_value = mock_session_cm

            client = MCPClient(
                mock_transport["transport_callable"],
                logging_callback=mock_logging,
            )
            with client:
                pass

            call_kwargs = MockSession.call_args[1]
            assert call_kwargs["logging_callback"] is mock_logging

    def test_default_logging_callback_used(self, mock_transport):
        """When no logging_callback provided, default should be used."""
        with patch("strands.tools.mcp.mcp_client.ClientSession") as MockSession:
            mock_session_instance = AsyncMock()
            mock_init_result = MagicMock()
            mock_init_result.instructions = None
            mock_session_instance.initialize = AsyncMock(return_value=mock_init_result)
            mock_session_instance.get_server_capabilities = MagicMock(return_value=None)

            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__.return_value = mock_session_instance
            MockSession.return_value = mock_session_cm

            client = MCPClient(mock_transport["transport_callable"])
            with client:
                pass

            call_kwargs = MockSession.call_args[1]
            assert call_kwargs["logging_callback"] is _default_logging_callback

    def test_progress_callback_passed_to_call_tool(self, mock_transport, mock_session):
        """progress_callback should be passed to session.call_tool()."""
        mock_progress = MagicMock()
        mock_call_result = MCPCallToolResult(
            content=[TextContent(type="text", text="ok")],
            isError=False,
        )
        mock_session.call_tool = AsyncMock(return_value=mock_call_result)

        client = MCPClient(
            mock_transport["transport_callable"],
            progress_callback=mock_progress,
        )
        with client:
            client.call_tool_sync("tool-use-prog", "my_tool", {"x": 1})

        call_kwargs = mock_session.call_tool.call_args[1]
        assert call_kwargs["progress_callback"] is mock_progress

    def test_none_logging_callback_disables_logging(self, mock_transport):
        """Passing logging_callback=None should disable log handling."""
        with patch("strands.tools.mcp.mcp_client.ClientSession") as MockSession:
            mock_session_instance = AsyncMock()
            mock_init_result = MagicMock()
            mock_init_result.instructions = None
            mock_session_instance.initialize = AsyncMock(return_value=mock_init_result)
            mock_session_instance.get_server_capabilities = MagicMock(return_value=None)

            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__.return_value = mock_session_instance
            MockSession.return_value = mock_session_cm

            client = MCPClient(
                mock_transport["transport_callable"],
                logging_callback=None,
            )
            with client:
                pass

            call_kwargs = MockSession.call_args[1]
            assert call_kwargs["logging_callback"] is None


# ==================================================================================
# Default Logging Callback Tests
# ==================================================================================


class TestDefaultLoggingCallback:
    """Test the default logging callback routes MCP logs to Python logging."""

    def test_info_level(self):
        """MCP 'info' level should map to Python INFO."""
        params = LoggingMessageNotificationParams(level="info", data="test message")
        with patch("strands.tools.mcp.mcp_client.logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            _default_logging_callback(params)

            mock_get_logger.assert_called_with("strands.mcp.server")
            mock_logger.log.assert_called_once_with(logging.INFO, "%s", "test message")

    def test_error_level(self):
        """MCP 'error' level should map to Python ERROR."""
        params = LoggingMessageNotificationParams(level="error", data="oops")
        with patch("strands.tools.mcp.mcp_client.logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            _default_logging_callback(params)

            mock_logger.log.assert_called_once_with(logging.ERROR, "%s", "oops")

    def test_critical_level(self):
        """MCP 'critical' level should map to Python CRITICAL."""
        params = LoggingMessageNotificationParams(level="critical", data="bad")
        with patch("strands.tools.mcp.mcp_client.logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            _default_logging_callback(params)

            mock_logger.log.assert_called_once_with(logging.CRITICAL, "%s", "bad")

    def test_named_logger(self):
        """MCP logger name should be included in Python logger name."""
        params = LoggingMessageNotificationParams(
            level="info", logger="my-module", data="test"
        )
        with patch("strands.tools.mcp.mcp_client.logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            _default_logging_callback(params)

            mock_get_logger.assert_called_with("strands.mcp.server.my-module")

    def test_notice_level_maps_to_info(self):
        """MCP 'notice' level should map to Python INFO."""
        params = LoggingMessageNotificationParams(level="notice", data="low level")
        with patch("strands.tools.mcp.mcp_client.logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            _default_logging_callback(params)

            mock_logger.log.assert_called_once_with(logging.INFO, "%s", "low level")


# ==================================================================================
# Config Loading Tests
# ==================================================================================


class TestConfigLoading:
    """Test load_mcp_servers() configuration loading."""

    def test_load_from_dict(self):
        """Should create MCPClient instances from a dict config."""
        config = {
            "mcpServers": {
                "test-server": {
                    "command": "echo",
                    "args": ["hello"],
                }
            }
        }
        clients = load_mcp_servers(config)
        assert len(clients) == 1
        assert isinstance(clients[0], MCPClient)

    def test_load_from_json_string(self):
        """Should create MCPClient instances from a JSON string."""
        config_str = json.dumps({
            "mcpServers": {
                "test-server": {
                    "command": "echo",
                    "args": ["hello"],
                }
            }
        })
        clients = load_mcp_servers(config_str)
        assert len(clients) == 1

    def test_load_from_file(self):
        """Should create MCPClient instances from a JSON file."""
        config = {
            "mcpServers": {
                "test-server": {
                    "command": "echo",
                    "args": ["hello"],
                }
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            f.flush()

            clients = load_mcp_servers(f.name)
            assert len(clients) == 1

    def test_disabled_server_skipped(self):
        """Disabled servers should be skipped."""
        config = {
            "mcpServers": {
                "active": {"command": "echo", "args": ["hi"]},
                "inactive": {"command": "echo", "args": ["bye"], "disabled": True},
            }
        }
        clients = load_mcp_servers(config)
        assert len(clients) == 1

    def test_disabled_tools_create_filters(self):
        """disabledTools should create tool filters."""
        config = {
            "mcpServers": {
                "test": {
                    "command": "echo",
                    "args": [],
                    "disabledTools": ["bad_tool"],
                }
            }
        }
        clients = load_mcp_servers(config)
        assert len(clients) == 1
        assert clients[0]._tool_filters is not None
        assert clients[0]._tool_filters["rejected"] == ["bad_tool"]

    def test_prefix_defaults_to_server_name(self):
        """prefix should default to the server name key."""
        config = {
            "mcpServers": {
                "my-server": {"command": "echo", "args": []}
            }
        }
        clients = load_mcp_servers(config)
        assert clients[0]._prefix == "my-server"

    def test_custom_prefix(self):
        """Custom prefix should override default."""
        config = {
            "mcpServers": {
                "my-server": {"command": "echo", "args": [], "prefix": "custom"}
            }
        }
        clients = load_mcp_servers(config)
        assert clients[0]._prefix == "custom"

    def test_url_based_server(self):
        """URL-based servers should be loaded."""
        config = {
            "mcpServers": {
                "remote": {"url": "https://example.com/mcp"}
            }
        }
        clients = load_mcp_servers(config)
        assert len(clients) == 1

    def test_no_transport_skipped(self):
        """Servers with no command or url should be skipped."""
        config = {
            "mcpServers": {
                "empty": {"args": ["something"]}
            }
        }
        clients = load_mcp_servers(config)
        assert len(clients) == 0

    def test_empty_config(self):
        """Empty mcpServers should return empty list."""
        clients = load_mcp_servers({"mcpServers": {}})
        assert clients == []

    def test_callbacks_passed_to_all_clients(self):
        """Shared callbacks should be passed to all created clients."""
        mock_progress = MagicMock()
        config = {
            "mcpServers": {
                "s1": {"command": "echo", "args": []},
                "s2": {"command": "echo", "args": []},
            }
        }
        clients = load_mcp_servers(config, progress_callback=mock_progress)
        assert len(clients) == 2
        assert clients[0]._progress_callback is mock_progress
        assert clients[1]._progress_callback is mock_progress

    def test_invalid_json_string_raises(self):
        """Invalid JSON string should raise ValueError."""
        with pytest.raises(ValueError, match="config is not a valid file path or JSON string"):
            load_mcp_servers("not valid json {{{")

    def test_parse_config_dict(self):
        """_parse_config should pass through dicts."""
        result = _parse_config({"key": "value"})
        assert result == {"key": "value"}

    def test_parse_config_path(self):
        """_parse_config should handle Path objects."""
        config = {"mcpServers": {}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            f.flush()

            result = _parse_config(Path(f.name))
            assert result == config

    def test_load_from_path_object(self):
        """Should accept Path objects."""
        config = {
            "mcpServers": {
                "test": {"command": "echo", "args": []}
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            f.flush()

            clients = load_mcp_servers(Path(f.name))
            assert len(clients) == 1


# ==================================================================================
# MCP Events Tests
# ==================================================================================


class TestMCPEvents:
    """Test MCP event hierarchy and registration."""

    def test_server_connected_event(self):
        """MCPServerConnectedEvent should be creatable and inspectable."""
        event = MCPServerConnectedEvent(
            server_name="test-server",
            server_instructions="Be helpful",
            capabilities={"tools": True},
        )
        assert event.server_name == "test-server"
        assert event.server_instructions == "Be helpful"
        assert event.capabilities == {"tools": True}

    def test_server_disconnected_event_reverse_callbacks(self):
        """MCPServerDisconnectedEvent should use reverse callback ordering."""
        event = MCPServerDisconnectedEvent(server_name="test-server")
        assert event.should_reverse_callbacks is True

    def test_tool_call_start_event(self):
        """MCPToolCallStartEvent should capture tool call details."""
        event = MCPToolCallStartEvent(
            tool_name="my_tool",
            tool_use_id="abc-123",
            arguments={"x": 1},
            meta={"progressToken": "tok"},
        )
        assert event.tool_name == "my_tool"
        assert event.tool_use_id == "abc-123"
        assert event.arguments == {"x": 1}
        assert event.meta == {"progressToken": "tok"}

    def test_tool_call_end_event_reverse_callbacks(self):
        """MCPToolCallEndEvent should use reverse callback ordering."""
        event = MCPToolCallEndEvent(tool_name="my_tool", tool_use_id="abc-123")
        assert event.should_reverse_callbacks is True

    def test_logging_event(self):
        """MCPLoggingEvent should capture log details."""
        event = MCPLoggingEvent(
            level="error",
            logger_name="my-module",
            data="bad things",
            server_name="test-server",
        )
        assert event.level == "error"
        assert event.data == "bad things"

    def test_progress_event(self):
        """MCPProgressEvent should capture progress details."""
        event = MCPProgressEvent(
            tool_name="slow_tool",
            progress=50.0,
            total=100.0,
            message="halfway there",
        )
        assert event.progress == 50.0
        assert event.total == 100.0

    def test_sampling_request_event(self):
        """MCPSamplingRequestEvent should capture sampling request details."""
        event = MCPSamplingRequestEvent(
            server_name="test-server",
            model="claude-3",
            max_tokens=1000,
        )
        assert event.server_name == "test-server"

    def test_events_register_with_hook_registry(self):
        """All MCP events should be registerable with HookRegistry."""
        registry = HookRegistry()

        callback = MagicMock()
        registry.add_callback(MCPServerConnectedEvent, callback)
        registry.add_callback(MCPServerDisconnectedEvent, callback)
        registry.add_callback(MCPToolCallStartEvent, callback)
        registry.add_callback(MCPToolCallEndEvent, callback)
        registry.add_callback(MCPLoggingEvent, callback)
        registry.add_callback(MCPProgressEvent, callback)
        registry.add_callback(MCPSamplingRequestEvent, callback)

        assert registry.has_callbacks()

    def test_events_invoke_through_registry(self):
        """MCP events should be invokable through HookRegistry."""
        registry = HookRegistry()
        captured = []

        def on_connected(event: MCPServerConnectedEvent) -> None:
            captured.append(event)

        registry.add_callback(MCPServerConnectedEvent, on_connected)

        event = MCPServerConnectedEvent(server_name="test")
        registry.invoke_callbacks(event)

        assert len(captured) == 1
        assert captured[0].server_name == "test"


# ==================================================================================
# MCPToolResult Type Tests
# ==================================================================================


class TestMCPToolResultType:
    """Test MCPToolResult TypedDict has isError field."""

    def test_is_error_field_exists(self):
        """MCPToolResult should accept isError field."""
        result = MCPToolResult(
            status="error",
            toolUseId="test",
            content=[{"text": "error"}],
            isError=True,
        )
        assert result["isError"] is True

    def test_is_error_not_required(self):
        """MCPToolResult should work without isError (backward compat)."""
        result = MCPToolResult(
            status="success",
            toolUseId="test",
            content=[{"text": "ok"}],
        )
        assert "isError" not in result
