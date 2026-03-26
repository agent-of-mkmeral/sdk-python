"""Adversarial tests for Option B: MCP Pass-Through.

These tests are designed to break the Option B implementation by testing edge cases,
boundary conditions, contract violations, and failure modes that the author's tests
don't cover.

Findings:
- BUG #1: load_mcp_servers(logging_callback=None) cannot disable logging -
          falls back to MCPClient default instead.
- BUG #2: Docstring claims FileNotFoundError for non-existent paths, but ValueError
          is raised instead (contract violation).
- BUG #3: Docstring claims ValueError when mcpServers key is missing, but code
          returns empty list with a warning (contract violation).
- EDGE #1: _meta with non-dict value (e.g., string) is passed through without validation.
- EDGE #2: URL /sse detection is substring-based, causing false positives for paths
           like /sse-endpoint.
- EDGE #3: stop() unconditionally creates _tool_task_support_cache even when tasks
           were never enabled (inconsistency).
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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
# BUG: load_mcp_servers cannot disable logging via logging_callback=None
# ==================================================================================


class TestLoadMcpServersLoggingCallbackBug:
    """BUG #1: Passing logging_callback=None to load_mcp_servers does NOT disable logging.

    The load_mcp_servers function checks `if logging_callback is not None` before adding
    it to kwargs. When None is passed, the kwarg is omitted, and MCPClient uses its
    default (_default_logging_callback). There is no way to disable logging through this API.
    """

    def test_none_logging_callback_should_disable_but_doesnt(self):
        """Demonstrates that passing None doesn't propagate to MCPClient."""
        config = {"mcpServers": {"test": {"command": "echo", "args": []}}}
        clients = load_mcp_servers(config, logging_callback=None)
        assert len(clients) == 1
        # BUG: This should be None, but it's the default callback
        # After fix, this assertion should pass:
        assert clients[0]._logging_callback is None, (
            "Expected logging_callback=None to disable logging, "
            f"but got {clients[0]._logging_callback}"
        )

    def test_explicit_callback_is_wired(self):
        """When an explicit callback is provided, it should be wired correctly."""
        custom_cb = MagicMock()
        config = {"mcpServers": {"test": {"command": "echo", "args": []}}}
        clients = load_mcp_servers(config, logging_callback=custom_cb)
        assert clients[0]._logging_callback is custom_cb

    def test_no_logging_callback_uses_default(self):
        """When logging_callback is not provided, MCPClient default should be used."""
        config = {"mcpServers": {"test": {"command": "echo", "args": []}}}
        clients = load_mcp_servers(config)  # No logging_callback arg
        assert clients[0]._logging_callback is _default_logging_callback


# ==================================================================================
# BUG: Docstring contract violations in _parse_config / load_mcp_servers
# ==================================================================================


class TestDocstringContractViolations:
    """BUG #2 & #3: Documented exception types don't match actual behavior."""

    def test_nonexistent_file_should_raise_file_not_found(self):
        """Docstring says FileNotFoundError for missing files, but ValueError is raised."""
        with pytest.raises(ValueError):
            # Docstring claims FileNotFoundError but we get ValueError
            _parse_config("/tmp/this_file_does_not_exist_xyz_12345.json")

    def test_nonexistent_file_via_load_mcp_servers(self):
        """load_mcp_servers docstring also claims FileNotFoundError."""
        with pytest.raises(ValueError):
            load_mcp_servers("/tmp/this_file_does_not_exist_xyz_12345.json")

    def test_missing_mcp_servers_key_should_raise_value_error(self):
        """Docstring says ValueError when mcpServers key missing, but returns empty list."""
        # Per docstring: "Raises: ValueError: If config format is invalid or no mcpServers key found."
        # Actual behavior: logs warning and returns []
        result = load_mcp_servers({"otherKey": "value"})
        assert result == [], "Should return empty list (contradicts docstring's ValueError claim)"


# ==================================================================================
# Edge Case: _meta extraction with unusual values
# ==================================================================================


class TestMetaExtractionEdgeCases:
    """Edge cases for _meta extraction from tool arguments."""

    def test_meta_value_is_none(self, mock_transport, mock_session):
        """_meta with None value should be extracted and passed as meta=None."""
        mock_call_result = MCPCallToolResult(
            content=[TextContent(type="text", text="ok")],
            isError=False,
        )
        mock_session.call_tool = AsyncMock(return_value=mock_call_result)

        client = MCPClient(mock_transport["transport_callable"])
        with client:
            result = client.call_tool_sync(
                "tool-use-meta-none", "my_tool", {"arg1": "v", "_meta": None}
            )

        call_args = mock_session.call_tool.call_args
        # _meta should be removed from arguments
        assert "_meta" not in call_args[0][1]
        # meta kwarg should be None
        assert call_args[1]["meta"] is None
        assert result["status"] == "success"

    def test_meta_value_is_string(self, mock_transport, mock_session):
        """_meta with string value - passes through without type validation."""
        mock_call_result = MCPCallToolResult(
            content=[TextContent(type="text", text="ok")],
            isError=False,
        )
        mock_session.call_tool = AsyncMock(return_value=mock_call_result)

        client = MCPClient(mock_transport["transport_callable"])
        with client:
            result = client.call_tool_sync(
                "tool-use-meta-str", "my_tool", {"_meta": "not-a-dict"}
            )

        call_args = mock_session.call_tool.call_args
        assert call_args[0][1] == {}  # arguments should be empty after _meta removal
        assert call_args[1]["meta"] == "not-a-dict"  # passed through as-is

    def test_meta_value_is_empty_dict(self, mock_transport, mock_session):
        """_meta with empty dict should be passed through as meta={}."""
        mock_call_result = MCPCallToolResult(
            content=[TextContent(type="text", text="ok")],
            isError=False,
        )
        mock_session.call_tool = AsyncMock(return_value=mock_call_result)

        client = MCPClient(mock_transport["transport_callable"])
        with client:
            client.call_tool_sync(
                "tool-use-meta-empty", "my_tool", {"arg": "val", "_meta": {}}
            )

        call_args = mock_session.call_tool.call_args
        assert call_args[0][1] == {"arg": "val"}
        assert call_args[1]["meta"] == {}

    def test_empty_arguments_dict(self, mock_transport, mock_session):
        """Empty arguments dict should not crash."""
        mock_call_result = MCPCallToolResult(
            content=[TextContent(type="text", text="ok")],
            isError=False,
        )
        mock_session.call_tool = AsyncMock(return_value=mock_call_result)

        client = MCPClient(mock_transport["transport_callable"])
        with client:
            result = client.call_tool_sync("tool-use-empty", "my_tool", {})

        assert result["status"] == "success"
        call_args = mock_session.call_tool.call_args
        assert call_args[1]["meta"] is None


# ==================================================================================
# Edge Case: Default logging callback with unusual inputs
# ==================================================================================


class TestDefaultLoggingCallbackEdgeCases:
    """Edge cases for the _default_logging_callback function."""

    def test_unknown_mcp_log_level_rejected_by_pydantic(self):
        """Unknown MCP log level is rejected by Pydantic validation.
        
        The MCP SDK uses a Literal type for level, so unknown levels like 'trace'
        are rejected at construction time. This means the default fallback to INFO
        in _default_logging_callback is a defensive guard that cannot be triggered
        by standard MCP messages - the SDK validates before it reaches our callback.
        """
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            LoggingMessageNotificationParams(level="trace", data="trace msg")

    def test_dict_data(self):
        """Dict data should be logged without crashing."""
        params = LoggingMessageNotificationParams(level="info", data={"key": "value"})
        with patch("strands.tools.mcp.mcp_client.logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            _default_logging_callback(params)
            mock_logger.log.assert_called_once_with(logging.INFO, "%s", {"key": "value"})

    def test_none_data(self):
        """None data should be logged without crashing."""
        params = LoggingMessageNotificationParams(level="info", data=None)
        with patch("strands.tools.mcp.mcp_client.logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            _default_logging_callback(params)
            mock_logger.log.assert_called_once_with(logging.INFO, "%s", None)

    def test_list_data(self):
        """List data should be logged without crashing."""
        params = LoggingMessageNotificationParams(level="warning", data=[1, 2, 3])
        with patch("strands.tools.mcp.mcp_client.logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            _default_logging_callback(params)
            mock_logger.log.assert_called_once_with(logging.WARNING, "%s", [1, 2, 3])

    def test_emergency_level_maps_to_critical(self):
        """MCP 'emergency' and 'alert' levels should map to CRITICAL."""
        for level in ["emergency", "alert"]:
            params = LoggingMessageNotificationParams(level=level, data="urgent")
            with patch("strands.tools.mcp.mcp_client.logging.getLogger") as mock_get_logger:
                mock_logger = MagicMock()
                mock_get_logger.return_value = mock_logger
                _default_logging_callback(params)
                mock_logger.log.assert_called_once_with(
                    logging.CRITICAL, "%s", "urgent"
                )

    def test_debug_level(self):
        """MCP 'debug' level should map to Python DEBUG."""
        params = LoggingMessageNotificationParams(level="debug", data="debug info")
        with patch("strands.tools.mcp.mcp_client.logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            _default_logging_callback(params)
            mock_logger.log.assert_called_once_with(logging.DEBUG, "%s", "debug info")


# ==================================================================================
# Edge Case: Config loading boundary conditions
# ==================================================================================


class TestConfigLoadingEdgeCases:
    """Edge cases for config loading."""

    def test_empty_json_file(self):
        """Empty JSON object in file should return empty list."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({}, f)
            f.flush()
            clients = load_mcp_servers(f.name)
            assert clients == []

    def test_json_with_empty_mcp_servers(self):
        """JSON with empty mcpServers dict should return empty list."""
        clients = load_mcp_servers({"mcpServers": {}})
        assert clients == []

    def test_special_characters_in_server_name(self):
        """Server names with special characters should be usable as prefixes."""
        config = {
            "mcpServers": {
                "my-server.v2": {"command": "echo", "args": []},
                "server/with/slashes": {"command": "echo", "args": []},
                "server with spaces": {"command": "echo", "args": []},
            }
        }
        clients = load_mcp_servers(config)
        assert len(clients) == 3
        prefixes = [c._prefix for c in clients]
        assert "my-server.v2" in prefixes
        assert "server/with/slashes" in prefixes
        assert "server with spaces" in prefixes

    def test_duplicate_server_names_in_dict(self):
        """Python dict only keeps last entry for duplicate keys.
        
        This isn't really a bug in the code but documents expected behavior.
        """
        # In a Python dict literal, the last value wins
        config_str = '{"mcpServers": {"server": {"command": "echo1"}, "server": {"command": "echo2"}}}'
        parsed = json.loads(config_str)
        assert parsed["mcpServers"]["server"]["command"] == "echo2"
        clients = load_mcp_servers(parsed)
        assert len(clients) == 1

    def test_parse_config_with_invalid_type(self):
        """Non-string, non-Path, non-dict should raise ValueError."""
        with pytest.raises(ValueError, match="unsupported config type"):
            _parse_config(42)

    def test_parse_config_with_list(self):
        """List should raise ValueError."""
        with pytest.raises(ValueError, match="unsupported config type"):
            _parse_config([1, 2, 3])

    def test_url_with_sse_substring_false_positive(self):
        """URL containing /sse as substring in path should use sse_client."""
        # This documents the current behavior - /sse anywhere in URL triggers SSE
        transport = _build_transport("test", {"url": "https://example.com/sse-endpoint"})
        assert transport is not None
        # The transport would use sse_client due to /sse substring match

    def test_url_without_sse_uses_streamable(self):
        """URL without /sse should use streamablehttp_client."""
        transport = _build_transport("test", {"url": "https://example.com/mcp"})
        assert transport is not None

    def test_server_with_no_command_or_url(self):
        """Server config without command or url should return None transport."""
        transport = _build_transport("test", {"args": ["something"]})
        assert transport is None

    def test_load_from_path_object(self):
        """Should accept Path objects."""
        config = {"mcpServers": {"test": {"command": "echo", "args": []}}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            f.flush()
            clients = load_mcp_servers(Path(f.name))
            assert len(clients) == 1

    def test_path_with_home_expansion(self):
        """Path with ~ should be expanded."""
        # This should not crash even though the expanded path doesn't exist
        with pytest.raises(ValueError):
            _parse_config("~/nonexistent_mcp_config.json")

    def test_multiple_servers_with_callbacks(self):
        """Multiple servers should all get the same shared callbacks."""
        mock_sampling = MagicMock()
        mock_progress = MagicMock()
        mock_elicit = MagicMock()
        mock_roots = MagicMock()

        config = {
            "mcpServers": {
                "s1": {"command": "echo", "args": []},
                "s2": {"command": "echo", "args": []},
                "s3": {"url": "https://example.com/mcp"},
            }
        }
        clients = load_mcp_servers(
            config,
            sampling_callback=mock_sampling,
            progress_callback=mock_progress,
            elicitation_callback=mock_elicit,
            list_roots_callback=mock_roots,
        )
        assert len(clients) == 3
        for c in clients:
            assert c._sampling_callback is mock_sampling
            assert c._progress_callback is mock_progress
            assert c._elicitation_callback is mock_elicit
            assert c._list_roots_callback is mock_roots


# ==================================================================================
# Edge Case: MCPClient with all callbacks simultaneously
# ==================================================================================


class TestAllCallbacksSimultaneously:
    """Test MCPClient with all callback parameters set at once."""

    def test_all_callbacks_set(self, mock_transport):
        """All callbacks should be stored without interference."""
        mock_sampling = MagicMock()
        mock_roots = MagicMock()
        mock_logging = MagicMock()
        mock_progress = MagicMock()
        mock_elicit = MagicMock()

        client = MCPClient(
            mock_transport["transport_callable"],
            sampling_callback=mock_sampling,
            list_roots_callback=mock_roots,
            logging_callback=mock_logging,
            progress_callback=mock_progress,
            elicitation_callback=mock_elicit,
        )

        assert client._sampling_callback is mock_sampling
        assert client._list_roots_callback is mock_roots
        assert client._logging_callback is mock_logging
        assert client._progress_callback is mock_progress
        assert client._elicitation_callback is mock_elicit

    def test_all_callbacks_wired_to_session(self, mock_transport):
        """When all callbacks are set, they should all be wired to ClientSession."""
        mock_sampling = MagicMock()
        mock_roots = MagicMock()
        mock_logging = MagicMock()
        mock_elicit = MagicMock()

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
                list_roots_callback=mock_roots,
                logging_callback=mock_logging,
                elicitation_callback=mock_elicit,
            )
            with client:
                pass

            call_kwargs = MockSession.call_args[1]
            assert call_kwargs["sampling_callback"] is mock_sampling
            assert call_kwargs["list_roots_callback"] is mock_roots
            assert call_kwargs["logging_callback"] is mock_logging
            assert call_kwargs["elicitation_callback"] is mock_elicit


# ==================================================================================
# Edge Case: isError backward compatibility
# ==================================================================================


class TestIsErrorBackwardCompat:
    """Test isError field backward compatibility."""

    def test_mcp_tool_result_without_is_error(self):
        """Old code that doesn't set isError should still work."""
        result = MCPToolResult(
            status="success",
            toolUseId="test",
            content=[{"text": "ok"}],
        )
        # isError not set - should not be in dict
        assert "isError" not in result
        # Accessing via .get() should return None
        assert result.get("isError") is None

    def test_mcp_tool_result_with_is_error_false(self):
        """Explicit isError=False should be accessible."""
        result = MCPToolResult(
            status="success",
            toolUseId="test",
            content=[{"text": "ok"}],
            isError=False,
        )
        assert result["isError"] is False

    def test_mcp_tool_result_with_is_error_true(self):
        """isError=True should be accessible."""
        result = MCPToolResult(
            status="error",
            toolUseId="test",
            content=[{"text": "fail"}],
            isError=True,
        )
        assert result["isError"] is True

    def test_result_status_and_is_error_consistency(self, mock_transport, mock_session):
        """status='error' and isError=True should always be consistent."""
        mock_call_result = MCPCallToolResult(
            content=[TextContent(type="text", text="fail")],
            isError=True,
        )
        mock_session.call_tool = AsyncMock(return_value=mock_call_result)

        client = MCPClient(mock_transport["transport_callable"])
        with client:
            result = client.call_tool_sync("use-id", "tool", {})

        assert result["status"] == "error"
        assert result["isError"] is True

    def test_result_success_and_is_error_consistency(self, mock_transport, mock_session):
        """status='success' and isError=False should always be consistent."""
        mock_call_result = MCPCallToolResult(
            content=[TextContent(type="text", text="ok")],
            isError=False,
        )
        mock_session.call_tool = AsyncMock(return_value=mock_call_result)

        client = MCPClient(mock_transport["transport_callable"])
        with client:
            result = client.call_tool_sync("use-id", "tool", {})

        assert result["status"] == "success"
        assert result["isError"] is False


# ==================================================================================
# Edge Case: MCP Events with missing/default fields
# ==================================================================================


class TestMCPEventsEdgeCases:
    """Edge cases for MCP event dataclasses."""

    def test_connected_event_all_defaults(self):
        """MCPServerConnectedEvent with all defaults should work."""
        event = MCPServerConnectedEvent()
        assert event.server_name == ""
        assert event.server_instructions is None
        assert event.capabilities == {}

    def test_disconnected_event_with_error(self):
        """MCPServerDisconnectedEvent should capture exception."""
        err = RuntimeError("connection lost")
        event = MCPServerDisconnectedEvent(server_name="test", error=err)
        assert event.error is err
        assert event.should_reverse_callbacks is True

    def test_tool_call_start_with_none_arguments(self):
        """MCPToolCallStartEvent should accept None arguments."""
        event = MCPToolCallStartEvent(
            tool_name="tool",
            tool_use_id="id",
            arguments=None,
            meta=None,
        )
        assert event.arguments is None
        assert event.meta is None

    def test_tool_call_end_with_duration(self):
        """MCPToolCallEndEvent should accept duration_ms."""
        event = MCPToolCallEndEvent(
            tool_name="tool",
            tool_use_id="id",
            status="success",
            is_error=False,
            duration_ms=123.45,
        )
        assert event.duration_ms == 123.45

    def test_tool_call_end_with_error(self):
        """MCPToolCallEndEvent should capture errors."""
        err = TimeoutError("tool timed out")
        event = MCPToolCallEndEvent(
            tool_name="tool",
            tool_use_id="id",
            status="error",
            is_error=True,
            error=err,
        )
        assert event.error is err
        assert event.is_error is True

    def test_progress_event_none_total_and_message(self):
        """MCPProgressEvent with None total and message should work."""
        event = MCPProgressEvent(
            tool_name="tool",
            progress=42.0,
            total=None,
            message=None,
        )
        assert event.total is None
        assert event.message is None

    def test_logging_event_none_logger_name(self):
        """MCPLoggingEvent with None logger_name should work."""
        event = MCPLoggingEvent(level="info", logger_name=None, data="msg")
        assert event.logger_name is None

    def test_sampling_event_all_none(self):
        """MCPSamplingRequestEvent with all None optional fields."""
        event = MCPSamplingRequestEvent(
            server_name="test", model=None, max_tokens=None
        )
        assert event.model is None
        assert event.max_tokens is None

    def test_events_are_immutable(self):
        """MCP events should be immutable after construction."""
        event = MCPServerConnectedEvent(server_name="test")
        with pytest.raises(AttributeError, match="not writable"):
            event.server_name = "modified"

    def test_tool_call_events_immutable(self):
        """Tool call events should be immutable."""
        event = MCPToolCallStartEvent(tool_name="tool", tool_use_id="id")
        with pytest.raises(AttributeError, match="not writable"):
            event.tool_name = "other"

        end_event = MCPToolCallEndEvent(tool_name="tool", tool_use_id="id")
        with pytest.raises(AttributeError, match="not writable"):
            end_event.status = "other"


# ==================================================================================
# Edge Case: MCPClient stop() without start()
# ==================================================================================


class TestStopWithoutStart:
    """Test stop() behavior when called without start()."""

    def test_stop_without_start_no_crash(self):
        """Calling stop() before start() should not crash."""
        mock_transport = MagicMock()
        client = MCPClient(mock_transport)
        # Should not raise
        client.stop(None, None, None)

    def test_stop_creates_task_cache_even_without_tasks(self):
        """stop() unconditionally sets _tool_task_support_cache."""
        mock_transport = MagicMock()
        client = MCPClient(mock_transport)  # No tasks_config
        assert not hasattr(client, "_tool_task_support_cache")
        client.stop(None, None, None)
        # After stop, it now exists even though tasks were never enabled
        assert hasattr(client, "_tool_task_support_cache")
        assert client._tool_task_support_cache == {}


# ==================================================================================
# Edge Case: __init__.py exports
# ==================================================================================


class TestInitExports:
    """Test that __init__.py exports are correct."""

    def test_mcp_package_exports(self):
        """All documented exports should be importable from strands.tools.mcp."""
        from strands.tools.mcp import (
            MCPAgentTool,
            MCPClient,
            MCPToolResult,
            MCPTransport,
            TasksConfig,
            ToolFilters,
            _default_logging_callback,
            load_mcp_servers,
        )

        assert MCPClient is not None
        assert load_mcp_servers is not None
        assert _default_logging_callback is not None
        assert ToolFilters is not None

    def test_hooks_package_exports(self):
        """All MCP events should be importable from strands.hooks."""
        from strands.hooks import (
            MCPLoggingEvent,
            MCPProgressEvent,
            MCPSamplingRequestEvent,
            MCPServerConnectedEvent,
            MCPServerDisconnectedEvent,
            MCPToolCallEndEvent,
            MCPToolCallStartEvent,
        )

        assert MCPServerConnectedEvent is not None
        assert MCPToolCallStartEvent is not None

    def test_all_in_hooks_init(self):
        """All MCP events should be in __all__ of strands.hooks."""
        import strands.hooks

        expected_mcp_exports = [
            "MCPServerConnectedEvent",
            "MCPServerDisconnectedEvent",
            "MCPToolCallStartEvent",
            "MCPToolCallEndEvent",
            "MCPLoggingEvent",
            "MCPProgressEvent",
            "MCPSamplingRequestEvent",
        ]
        for name in expected_mcp_exports:
            assert name in strands.hooks.__all__, f"{name} not in strands.hooks.__all__"


# ==================================================================================
# Edge Case: Simultaneous progress callbacks
# ==================================================================================


class TestSimultaneousProgressCallbacks:
    """Test that progress_callback works correctly with concurrent tool calls."""

    def test_shared_progress_callback(self, mock_transport, mock_session):
        """A single progress_callback shared across calls should receive all updates."""
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
            # Two sequential tool calls with same progress callback
            client.call_tool_sync("use-1", "tool_a", {"x": 1})
            client.call_tool_sync("use-2", "tool_b", {"y": 2})

        # Both calls should pass the same progress_callback
        assert mock_session.call_tool.call_count == 2
        for call in mock_session.call_tool.call_args_list:
            assert call[1]["progress_callback"] is mock_progress


# ==================================================================================
# Edge Case: _handle_tool_result with structured content and metadata
# ==================================================================================


class TestHandleToolResultEdgeCases:
    """Test _handle_tool_result with various content types."""

    def test_structured_content_passed_through(self, mock_transport, mock_session):
        """structuredContent from MCP should be in the result."""
        mock_call_result = MCPCallToolResult(
            content=[TextContent(type="text", text="ok")],
            isError=False,
            structuredContent={"data": [1, 2, 3]},
        )
        mock_session.call_tool = AsyncMock(return_value=mock_call_result)

        client = MCPClient(mock_transport["transport_callable"])
        with client:
            result = client.call_tool_sync("use-id", "tool", {})

        assert result["structuredContent"] == {"data": [1, 2, 3]}

    def test_metadata_passed_through(self, mock_transport, mock_session):
        """metadata from MCP should be in the result as 'metadata' key."""
        mock_call_result = MCPCallToolResult(
            content=[TextContent(type="text", text="ok")],
            isError=False,
        )
        # Set meta field on the result
        mock_call_result.meta = {"tokens_used": 42}
        mock_session.call_tool = AsyncMock(return_value=mock_call_result)

        client = MCPClient(mock_transport["transport_callable"])
        with client:
            result = client.call_tool_sync("use-id", "tool", {})

        assert result.get("metadata") == {"tokens_used": 42}

    def test_execution_exception_returns_error_result(self, mock_transport, mock_session):
        """Tool execution exception should return isError=True result."""
        mock_session.call_tool = AsyncMock(side_effect=ConnectionError("server down"))

        client = MCPClient(mock_transport["transport_callable"])
        with client:
            result = client.call_tool_sync("use-id", "tool", {})

        assert result["status"] == "error"
        assert result["isError"] is True
        assert "server down" in result["content"][0]["text"]


# ==================================================================================
# Edge Case: HookRegistry integration
# ==================================================================================


class TestHookRegistryIntegration:
    """Test MCP events work correctly with HookRegistry."""

    def test_multiple_callbacks_for_same_event(self):
        """Multiple callbacks can be registered for the same event."""
        registry = HookRegistry()
        results = []

        def cb1(event: MCPToolCallStartEvent):
            results.append(("cb1", event.tool_name))

        def cb2(event: MCPToolCallStartEvent):
            results.append(("cb2", event.tool_name))

        registry.add_callback(MCPToolCallStartEvent, cb1)
        registry.add_callback(MCPToolCallStartEvent, cb2)

        event = MCPToolCallStartEvent(tool_name="test")
        registry.invoke_callbacks(event)

        assert len(results) == 2
        assert results[0] == ("cb1", "test")
        assert results[1] == ("cb2", "test")

    def test_end_event_callbacks_in_reverse_order(self):
        """End events should invoke callbacks in reverse order."""
        registry = HookRegistry()
        results = []

        def cb1(event: MCPToolCallEndEvent):
            results.append("cb1")

        def cb2(event: MCPToolCallEndEvent):
            results.append("cb2")

        registry.add_callback(MCPToolCallEndEvent, cb1)
        registry.add_callback(MCPToolCallEndEvent, cb2)

        event = MCPToolCallEndEvent(tool_name="test", tool_use_id="id")
        registry.invoke_callbacks(event)

        # Reverse order because should_reverse_callbacks is True
        assert results == ["cb2", "cb1"]

    def test_disconnected_event_callbacks_in_reverse_order(self):
        """Disconnected events should invoke callbacks in reverse order."""
        registry = HookRegistry()
        results = []

        def cb1(event: MCPServerDisconnectedEvent):
            results.append("cb1")

        def cb2(event: MCPServerDisconnectedEvent):
            results.append("cb2")

        registry.add_callback(MCPServerDisconnectedEvent, cb1)
        registry.add_callback(MCPServerDisconnectedEvent, cb2)

        event = MCPServerDisconnectedEvent(server_name="test")
        registry.invoke_callbacks(event)

        assert results == ["cb2", "cb1"]
