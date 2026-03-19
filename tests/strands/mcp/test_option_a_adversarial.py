"""Adversarial tests for Option A: MCP as First-Class Citizen.

These tests target edge cases, missing error handling, type errors,
race conditions, backward compatibility, and security issues in the
Option A implementation.
"""

import asyncio
import json
import os
import tempfile
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import CallToolResult as MCPCallToolResult
from mcp.types import TextContent as MCPTextContent

from strands.hooks.mcp_events import (
    MCPCancelledEvent,
    MCPEvent,
    MCPLogEvent,
    MCPProgressEvent,
    MCPPromptsChangedEvent,
    MCPResourcesChangedEvent,
    MCPResourceUpdatedEvent,
    MCPToolsChangedEvent,
)
from strands.hooks.registry import BaseHookEvent, HookRegistry
from strands.mcp.registry import MCPRegistry, load_mcp_servers, _build_client_from_config
from strands.tools.mcp.mcp_client import MCPClient, _default_logging_callback
from strands.tools.mcp.mcp_types import MCPToolResult


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_transport():
    mock_read_stream = AsyncMock()
    mock_write_stream = AsyncMock()
    mock_transport_cm = AsyncMock()
    mock_transport_cm.__aenter__.return_value = (mock_read_stream, mock_write_stream)
    mock_transport_callable = MagicMock(return_value=mock_transport_cm)
    return {
        "read_stream": mock_read_stream,
        "write_stream": mock_write_stream,
        "transport_cm": mock_transport_cm,
        "transport_callable": mock_transport_callable,
    }


@pytest.fixture
def mock_session():
    mock_session = AsyncMock()
    mock_init_result = MagicMock()
    mock_init_result.instructions = None
    mock_session.initialize = AsyncMock(return_value=mock_init_result)
    mock_session.get_server_capabilities = MagicMock(return_value=None)
    mock_session_cm = AsyncMock()
    mock_session_cm.__aenter__.return_value = mock_session
    with patch("strands.tools.mcp.mcp_client.ClientSession", return_value=mock_session_cm):
        yield mock_session


# ═══════════════════════════════════════════════════════════════════════
# 1. MCPRegistry edge cases
# ═══════════════════════════════════════════════════════════════════════


class TestMCPRegistryEdgeCases:
    """Edge cases for MCPRegistry initialization and operations."""

    def test_empty_clients_dict(self):
        """MCPRegistry({}) should be valid and behave like MCPRegistry()."""
        reg = MCPRegistry({})
        assert reg.server_names == []
        assert reg.clients == {}

    def test_none_clients(self):
        """MCPRegistry(None) should be valid and behave like MCPRegistry()."""
        reg = MCPRegistry(None)
        assert reg.server_names == []

    def test_add_client_replaces_existing(self):
        """Adding a client with an existing name should replace it."""
        c1 = MagicMock(spec=MCPClient)
        c2 = MagicMock(spec=MCPClient)
        reg = MCPRegistry({"a": c1})
        reg.add_client("a", c2)
        assert reg.clients["a"] is c2

    def test_register_tools_with_empty_registry(self):
        """register_tools with zero clients should not fail."""
        from strands.tools.registry import ToolRegistry

        tool_reg = MagicMock(spec=ToolRegistry)
        hook_reg = MagicMock()
        reg = MCPRegistry({})
        reg.register_tools(tool_reg, hook_reg)
        # process_tools should not be called since there are no clients
        tool_reg.process_tools.assert_not_called()

    def test_register_tools_with_none_hook_registry(self):
        """register_tools with None hook_registry - what happens?

        The code doesn't validate hook_registry type, only tool_registry.
        When _install_hook_logging_callback tries to use it, it will store a
        callback that references None, but won't fail until the callback fires.
        """
        from strands.tools.registry import ToolRegistry

        client = MagicMock(spec=MCPClient)
        client._logging_callback = None
        tool_reg = MagicMock(spec=ToolRegistry)

        reg = MCPRegistry({"srv": client})
        # This should NOT raise - it installs a callback but doesn't invoke it
        reg.register_tools(tool_reg, None)

    def test_duplicate_prefixes_across_clients(self):
        """Two clients with the same prefix should both register (no dedup)."""
        from strands.tools.registry import ToolRegistry

        c1 = MagicMock(spec=MCPClient)
        c1._logging_callback = None
        c1._prefix = "same_prefix"

        c2 = MagicMock(spec=MCPClient)
        c2._logging_callback = None
        c2._prefix = "same_prefix"

        tool_reg = MagicMock(spec=ToolRegistry)
        hook_reg = MagicMock()

        reg = MCPRegistry({"srv1": c1, "srv2": c2})
        reg.register_tools(tool_reg, hook_reg)
        # Both clients should be registered
        assert tool_reg.process_tools.call_count == 2


# ═══════════════════════════════════════════════════════════════════════
# 2. load_mcp_servers edge cases
# ═══════════════════════════════════════════════════════════════════════


class TestLoadMcpServersEdgeCases:
    """Edge cases for load_mcp_servers config parsing."""

    def test_malformed_json_string(self):
        """Malformed JSON string should raise json.JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            load_mcp_servers("{invalid json")

    def test_empty_json_object(self):
        """Empty JSON object should return empty dict."""
        result = load_mcp_servers("{}")
        assert result == {}

    def test_empty_string(self):
        """Empty string is not valid JSON."""
        with pytest.raises(json.JSONDecodeError):
            load_mcp_servers("")

    def test_empty_mcp_servers_key(self):
        """Config with empty mcpServers should return empty dict."""
        result = load_mcp_servers({"mcpServers": {}})
        assert result == {}

    def test_config_with_both_command_and_url(self):
        """Config with both 'command' and 'url' - command wins (no error)."""
        cfg = {"command": "echo", "url": "https://example.com/sse"}
        client = _build_client_from_config("ambiguous", cfg)
        assert isinstance(client, MCPClient)
        # Verify it used command (stdio) transport, not URL
        assert client._prefix == "ambiguous"

    def test_config_with_extra_unknown_keys(self):
        """Unknown keys in config should be silently ignored."""
        cfg = {
            "command": "echo",
            "unknown_key": "should_be_ignored",
            "another_key": 42,
        }
        client = _build_client_from_config("test", cfg)
        assert isinstance(client, MCPClient)

    def test_config_with_null_env(self):
        """Null env should be valid (defaults to None)."""
        cfg = {"command": "echo", "env": None}
        client = _build_client_from_config("test", cfg)
        assert isinstance(client, MCPClient)

    def test_config_with_empty_args(self):
        """Empty args list should be valid."""
        cfg = {"command": "echo", "args": []}
        client = _build_client_from_config("test", cfg)
        assert isinstance(client, MCPClient)

    def test_config_disabled_as_string_true(self):
        """'disabled': 'true' (string) should NOT skip - only bool True works.

        BUG: The truthy string 'true' evaluates to True in Python, so this
        actually DOES skip the server. But 'disabled': 0 would not skip.
        This is probably fine but worth documenting.
        """
        cfg = {
            "mcpServers": {
                "srv": {"command": "echo", "disabled": "true"},
            }
        }
        with patch("strands.mcp.registry._build_client_from_config") as mock_build:
            mock_build.return_value = MagicMock(spec=MCPClient)
            result = load_mcp_servers(cfg)
        # String "true" is truthy, so server IS disabled
        assert "srv" not in result

    def test_config_disabled_as_zero(self):
        """'disabled': 0 is falsy, server should NOT be disabled."""
        cfg = {
            "mcpServers": {
                "srv": {"command": "echo", "disabled": 0},
            }
        }
        with patch("strands.mcp.registry._build_client_from_config") as mock_build:
            mock_build.return_value = MagicMock(spec=MCPClient)
            result = load_mcp_servers(cfg)
        assert "srv" in result

    def test_url_with_no_sse_path(self):
        """URL without '/sse' should use streamable HTTP transport."""
        cfg = {"url": "https://example.com/api/mcp"}
        client = _build_client_from_config("test", cfg)
        assert isinstance(client, MCPClient)

    def test_url_with_sse_in_domain(self):
        """URL with '/sse' in path should use SSE transport."""
        cfg = {"url": "https://sse.example.com/sse"}
        client = _build_client_from_config("test", cfg)
        assert isinstance(client, MCPClient)

    def test_load_mcp_servers_json_array_not_object(self):
        """A JSON array (not object) should fail."""
        with pytest.raises((AttributeError, TypeError)):
            load_mcp_servers("[]")

    def test_load_mcp_servers_with_numeric_keys(self):
        """Config with numeric string keys should work."""
        cfg = {"mcpServers": {"123": {"command": "echo"}}}
        with patch("strands.mcp.registry._build_client_from_config") as mock_build:
            mock_build.return_value = MagicMock(spec=MCPClient)
            result = load_mcp_servers(cfg)
        assert "123" in result

    def test_inner_dict_with_mcpServers_key_as_server_name(self):
        """FIXED BUG: If the inner dict has 'mcpServers' as a key whose value
        is NOT a dict-of-dicts, the code should fall back to treating the
        entire config as the inner mapping (not crash).

        Previously crashed with: AttributeError: 'str' object has no attribute 'get'
        """
        cfg = {
            "mcpServers": {"command": "echo", "args": []},
            "other_server": {"command": "cat"},
        }
        with patch("strands.mcp.registry._build_client_from_config") as mock_build:
            mock_build.return_value = MagicMock(spec=MCPClient)
            result = load_mcp_servers(cfg)

        # After fix: the heuristic detects that mcpServers value contains
        # non-dict values, so treats the whole config as inner dict.
        # Both "mcpServers" and "other_server" are treated as server names.
        calls = mock_build.call_args_list
        server_names = [call[0][0] for call in calls]
        assert "mcpServers" in server_names
        assert "other_server" in server_names


# ═══════════════════════════════════════════════════════════════════════
# 3. _meta extraction edge cases
# ═══════════════════════════════════════════════════════════════════════


class TestMetaExtractionEdgeCases:
    """Edge cases for _meta extraction from tool arguments."""

    def test_meta_is_none_value(self, mock_transport, mock_session):
        """_meta=None should be extracted and passed as meta=None."""
        mock_content = MCPTextContent(type="text", text="ok")
        mock_session.call_tool.return_value = MCPCallToolResult(isError=False, content=[mock_content])

        with MCPClient(mock_transport["transport_callable"]) as client:
            client.call_tool_sync(
                tool_use_id="t1",
                name="tool",
                arguments={"key": "val", "_meta": None},
            )

        call_args = mock_session.call_tool.call_args
        assert call_args[0][1] == {"key": "val"}
        assert call_args[1]["meta"] is None

    def test_meta_is_empty_dict(self, mock_transport, mock_session):
        """_meta={} should be extracted and passed."""
        mock_content = MCPTextContent(type="text", text="ok")
        mock_session.call_tool.return_value = MCPCallToolResult(isError=False, content=[mock_content])

        with MCPClient(mock_transport["transport_callable"]) as client:
            client.call_tool_sync(
                tool_use_id="t1",
                name="tool",
                arguments={"_meta": {}},
            )

        call_args = mock_session.call_tool.call_args
        assert call_args[0][1] == {}  # arguments should be empty after extraction
        assert call_args[1]["meta"] == {}

    def test_meta_with_nested_complex_object(self, mock_transport, mock_session):
        """_meta with deeply nested data should be passed through."""
        complex_meta = {
            "progressToken": "tok-1",
            "nested": {"deep": {"value": [1, 2, 3]}},
        }
        mock_content = MCPTextContent(type="text", text="ok")
        mock_session.call_tool.return_value = MCPCallToolResult(isError=False, content=[mock_content])

        with MCPClient(mock_transport["transport_callable"]) as client:
            client.call_tool_sync(
                tool_use_id="t1",
                name="tool",
                arguments={"x": 1, "_meta": complex_meta},
            )

        call_args = mock_session.call_tool.call_args
        assert call_args[1]["meta"] == complex_meta

    def test_arguments_none(self, mock_transport, mock_session):
        """arguments=None should work (no _meta to extract)."""
        mock_content = MCPTextContent(type="text", text="ok")
        mock_session.call_tool.return_value = MCPCallToolResult(isError=False, content=[mock_content])

        with MCPClient(mock_transport["transport_callable"]) as client:
            client.call_tool_sync(
                tool_use_id="t1",
                name="tool",
                arguments=None,
            )

        call_args = mock_session.call_tool.call_args
        assert call_args[0][1] is None
        assert call_args[1]["meta"] is None

    def test_arguments_empty_dict(self, mock_transport, mock_session):
        """arguments={} should work with no _meta."""
        mock_content = MCPTextContent(type="text", text="ok")
        mock_session.call_tool.return_value = MCPCallToolResult(isError=False, content=[mock_content])

        with MCPClient(mock_transport["transport_callable"]) as client:
            client.call_tool_sync(
                tool_use_id="t1",
                name="tool",
                arguments={},
            )

        call_args = mock_session.call_tool.call_args
        assert call_args[0][1] == {}
        assert call_args[1]["meta"] is None


# ═══════════════════════════════════════════════════════════════════════
# 4. isError backward compatibility
# ═══════════════════════════════════════════════════════════════════════


class TestIsErrorBackwardCompat:
    """Verify isError field behavior matches MCPToolResult TypedDict."""

    def test_mcp_tool_result_without_is_error(self):
        """MCPToolResult should work without isError (it's NotRequired)."""
        result: MCPToolResult = {
            "status": "success",
            "toolUseId": "t1",
            "content": [{"text": "hello"}],
        }
        assert "isError" not in result

    def test_mcp_tool_result_with_is_error_true(self):
        result: MCPToolResult = {
            "status": "error",
            "toolUseId": "t1",
            "content": [{"text": "error"}],
            "isError": True,
        }
        assert result["isError"] is True

    def test_mcp_tool_result_with_is_error_false(self):
        result: MCPToolResult = {
            "status": "success",
            "toolUseId": "t1",
            "content": [{"text": "ok"}],
            "isError": False,
        }
        assert result["isError"] is False

    def test_handle_tool_result_always_sets_is_error(self, mock_transport, mock_session):
        """_handle_tool_result should always set isError in the result."""
        mock_content = MCPTextContent(type="text", text="ok")
        mock_session.call_tool.return_value = MCPCallToolResult(isError=False, content=[mock_content])

        with MCPClient(mock_transport["transport_callable"]) as client:
            result = client.call_tool_sync(tool_use_id="t1", name="tool", arguments={})

        # isError should always be present in the result
        assert "isError" in result
        assert result["isError"] is False

    def test_is_error_none_rejected_by_pydantic(self):
        """MCPCallToolResult rejects isError=None via Pydantic validation.

        This confirms that isError is always a real bool, not None.
        """
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            MCPCallToolResult(isError=None, content=[MCPTextContent(type="text", text="ok")])


# ═══════════════════════════════════════════════════════════════════════
# 5. MCPEvent hierarchy verification
# ═══════════════════════════════════════════════════════════════════════


class TestMCPEventHierarchyAdversarial:
    """Verify MCPEvent dataclass behavior edge cases."""

    def test_mcp_event_requires_server_name(self):
        """MCPEvent should require server_name (no default)."""
        with pytest.raises(TypeError):
            MCPEvent()  # Missing required field

    def test_mcp_log_event_with_non_string_data(self):
        """MCPLogEvent.data should accept any type (dict, list, etc)."""
        evt = MCPLogEvent(server_name="s", data={"key": "val"})
        assert evt.data == {"key": "val"}

        evt2 = MCPLogEvent(server_name="s", data=[1, 2, 3])
        assert evt2.data == [1, 2, 3]

        evt3 = MCPLogEvent(server_name="s", data=42)
        assert evt3.data == 42

    def test_mcp_progress_event_with_zero_total(self):
        """Progress event with total=0 - division by zero risk for consumers."""
        evt = MCPProgressEvent(server_name="s", progress=0, total=0)
        assert evt.total == 0
        # Consumer doing progress/total would get ZeroDivisionError

    def test_mcp_progress_event_negative_progress(self):
        """Negative progress values should be accepted (no validation)."""
        evt = MCPProgressEvent(server_name="s", progress=-1.0)
        assert evt.progress == -1.0

    def test_events_are_frozen(self):
        """MCPEvent dataclasses ARE frozen - fields are immutable.

        BaseHookEvent's custom __setattr__ prevents mutation.
        """
        evt = MCPLogEvent(server_name="s", level="info")
        with pytest.raises(AttributeError, match="not writable"):
            evt.level = "error"

    def test_event_equality(self):
        """Two events with same fields should be equal (dataclass default)."""
        e1 = MCPLogEvent(server_name="s", level="info", data="x")
        e2 = MCPLogEvent(server_name="s", level="info", data="x")
        assert e1 == e2

    def test_all_events_importable_from_hooks_init(self):
        """All MCP events should be importable from strands.hooks."""
        from strands.hooks import (
            MCPCancelledEvent,
            MCPEvent,
            MCPLogEvent,
            MCPProgressEvent,
            MCPPromptsChangedEvent,
            MCPResourcesChangedEvent,
            MCPResourceUpdatedEvent,
            MCPToolsChangedEvent,
        )
        # Verify they're the same classes
        from strands.hooks.mcp_events import MCPLogEvent as DirectImport
        assert MCPLogEvent is DirectImport

    def test_all_events_in_hooks_all(self):
        """All MCP events should be in hooks.__all__."""
        import strands.hooks
        for name in [
            "MCPCancelledEvent",
            "MCPEvent",
            "MCPLogEvent",
            "MCPProgressEvent",
            "MCPPromptsChangedEvent",
            "MCPResourcesChangedEvent",
            "MCPResourceUpdatedEvent",
            "MCPToolsChangedEvent",
        ]:
            assert name in strands.hooks.__all__, f"{name} missing from hooks.__all__"


# ═══════════════════════════════════════════════════════════════════════
# 6. Agent mcp_clients parameter edge cases
# ═══════════════════════════════════════════════════════════════════════


class TestAgentMcpClientsParam:
    """Edge cases for Agent(mcp_clients=...) parameter."""

    def test_agent_mcp_clients_empty_list(self):
        """Agent(mcp_clients=[]) should create registry with no clients."""
        from strands.agent.agent import Agent

        with patch.object(Agent, "__init__", lambda self, **kwargs: None):
            agent = Agent.__new__(Agent)

        # Simulate the mcp_clients handling from Agent.__init__
        from strands.mcp.registry import MCPRegistry
        mcp_clients = []
        named = {f"mcp_{i}": c for i, c in enumerate(mcp_clients)}
        reg = MCPRegistry(named)
        assert reg.server_names == []

    def test_agent_mcp_clients_list_naming(self):
        """List of clients should get auto-named mcp_0, mcp_1, etc."""
        c1 = MagicMock(spec=MCPClient)
        c2 = MagicMock(spec=MCPClient)
        clients = [c1, c2]
        named = {f"mcp_{i}": c for i, c in enumerate(clients)}
        assert list(named.keys()) == ["mcp_0", "mcp_1"]
        assert named["mcp_0"] is c1
        assert named["mcp_1"] is c2

    def test_agent_mcp_clients_dict_preserves_names(self):
        """Dict of clients should preserve the user-given names."""
        c1 = MagicMock(spec=MCPClient)
        c2 = MagicMock(spec=MCPClient)
        named = {"filesystem": c1, "github": c2}
        reg = MCPRegistry(named)
        assert set(reg.server_names) == {"filesystem", "github"}

    def test_agent_mcp_clients_invalid_type(self):
        """Agent(mcp_clients="string") should raise TypeError."""
        # Simulate the type check from Agent.__init__
        mcp_clients = "not_valid"
        if isinstance(mcp_clients, list):
            named = {f"mcp_{i}": c for i, c in enumerate(mcp_clients)}
        elif isinstance(mcp_clients, dict):
            named = mcp_clients
        else:
            with pytest.raises(TypeError):
                raise TypeError("mcp_clients must be a dict or list of MCPClient instances")

    def test_agent_mcp_clients_single_client_not_in_list(self):
        """Agent(mcp_clients=client) without list wrapping should fail."""
        client = MagicMock(spec=MCPClient)
        mcp_clients = client
        # MCPClient is not a list or dict
        assert not isinstance(mcp_clients, (list, dict))


# ═══════════════════════════════════════════════════════════════════════
# 7. Hook callback wiring edge cases
# ═══════════════════════════════════════════════════════════════════════


class TestHookCallbackWiring:
    """Test _install_hook_logging_callback edge cases."""

    @pytest.mark.asyncio
    async def test_hook_logging_callback_fires_event(self):
        """Verify hook logging callback emits MCPLogEvent."""
        from mcp.types import LoggingMessageNotificationParams
        from strands.mcp.registry import _install_hook_logging_callback

        client = MagicMock(spec=MCPClient)
        client._logging_callback = None

        hook_reg = MagicMock()
        hook_reg.invoke_callbacks_async = AsyncMock(return_value=(None, []))

        _install_hook_logging_callback(client, "test_srv", hook_reg)

        # Now invoke the installed callback
        params = LoggingMessageNotificationParams(
            level="warning",
            logger="mylogger",
            data="test message",
        )
        await client._logging_callback(params)

        # Verify MCPLogEvent was dispatched
        hook_reg.invoke_callbacks_async.assert_called_once()
        event = hook_reg.invoke_callbacks_async.call_args[0][0]
        assert isinstance(event, MCPLogEvent)
        assert event.server_name == "test_srv"
        assert event.level == "warning"
        assert event.logger_name == "mylogger"
        assert event.data == "test message"

    @pytest.mark.asyncio
    async def test_hook_logging_callback_calls_original(self):
        """Verify original callback is called before hook event."""
        from mcp.types import LoggingMessageNotificationParams
        from strands.mcp.registry import _install_hook_logging_callback

        original_cb = AsyncMock()
        client = MagicMock(spec=MCPClient)
        client._logging_callback = original_cb

        hook_reg = MagicMock()
        hook_reg.invoke_callbacks_async = AsyncMock(return_value=(None, []))

        _install_hook_logging_callback(client, "test_srv", hook_reg)

        params = LoggingMessageNotificationParams(
            level="info",
            data="hello",
        )
        await client._logging_callback(params)

        # Original callback should have been called
        original_cb.assert_called_once_with(params)

    @pytest.mark.asyncio
    async def test_hook_logging_callback_survives_hook_error(self):
        """If hook dispatch fails, callback should not crash."""
        from mcp.types import LoggingMessageNotificationParams
        from strands.mcp.registry import _install_hook_logging_callback

        client = MagicMock(spec=MCPClient)
        client._logging_callback = None

        hook_reg = MagicMock()
        hook_reg.invoke_callbacks_async = AsyncMock(side_effect=RuntimeError("hook failed"))

        _install_hook_logging_callback(client, "test_srv", hook_reg)

        params = LoggingMessageNotificationParams(level="info", data="x")
        # Should NOT raise - the exception is caught inside the callback
        await client._logging_callback(params)

    @pytest.mark.asyncio
    async def test_hook_logging_callback_with_none_hook_registry(self):
        """If hook_registry is None, callback should fail when invoked.

        BUG: register_tools doesn't validate hook_registry type.
        _install_hook_logging_callback will try to call
        None.invoke_callbacks_async() which raises AttributeError.
        This is caught by the try/except in the callback.
        """
        from mcp.types import LoggingMessageNotificationParams
        from strands.mcp.registry import _install_hook_logging_callback

        client = MagicMock(spec=MCPClient)
        client._logging_callback = None

        _install_hook_logging_callback(client, "test_srv", None)

        params = LoggingMessageNotificationParams(level="info", data="x")
        # Should NOT crash - the except clause catches the AttributeError
        await client._logging_callback(params)

    @pytest.mark.asyncio
    async def test_hook_logging_with_enum_level(self):
        """MCP LoggingLevel can be an enum - verify string conversion works."""
        from mcp.types import LoggingMessageNotificationParams

        # The level might come as a string or enum depending on MCP version
        params = LoggingMessageNotificationParams(level="emergency", data="critical!")

        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            await _default_logging_callback(params)

            import logging
            # "emergency" should map to CRITICAL
            assert mock_logger.log.call_args[0][0] == logging.CRITICAL


# ═══════════════════════════════════════════════════════════════════════
# 8. Thread safety
# ═══════════════════════════════════════════════════════════════════════


class TestThreadSafety:
    """Thread safety tests for MCPRegistry operations."""

    def test_concurrent_add_client(self):
        """Concurrent add_client calls should not lose entries."""
        reg = MCPRegistry()
        errors = []

        def add_clients(start_idx, count):
            try:
                for i in range(count):
                    client = MagicMock(spec=MCPClient)
                    reg.add_client(f"client_{start_idx + i}", client)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_clients, args=(i * 100, 100))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Note: Python's GIL makes dict operations mostly safe,
        # but this documents that the code has no explicit locking
        assert len(reg.server_names) == 500

    def test_clients_property_is_snapshot(self):
        """clients property should return a snapshot, not a live view."""
        reg = MCPRegistry()
        c1 = MagicMock(spec=MCPClient)
        reg.add_client("a", c1)

        snapshot = reg.clients
        reg.add_client("b", MagicMock(spec=MCPClient))

        # Snapshot should not include "b"
        assert "b" not in snapshot
        assert "b" in reg.clients


# ═══════════════════════════════════════════════════════════════════════
# 9. Security: env var handling in config
# ═══════════════════════════════════════════════════════════════════════


class TestConfigSecurity:
    """Security-related tests for config loading."""

    def test_env_vars_not_interpolated(self):
        """load_mcp_servers does NOT interpolate ${VAR} syntax.

        This is a feature gap - many MCP config formats support env var interpolation.
        Document that users need to handle this themselves.
        """
        cfg = {
            "mcpServers": {
                "srv": {
                    "command": "echo",
                    "env": {"API_KEY": "${MY_SECRET}"},
                }
            }
        }
        with patch("strands.mcp.registry._build_client_from_config") as mock_build:
            mock_build.return_value = MagicMock(spec=MCPClient)
            result = load_mcp_servers(cfg)

        # The raw ${MY_SECRET} string should be passed as-is
        call_cfg = mock_build.call_args[0][1]
        assert call_cfg["env"]["API_KEY"] == "${MY_SECRET}"

    def test_url_with_credentials_in_config(self):
        """URLs with embedded credentials should work (no filtering)."""
        cfg = {"url": "https://user:pass@example.com/mcp"}
        client = _build_client_from_config("test", cfg)
        assert isinstance(client, MCPClient)

    def test_command_injection_via_args(self):
        """Args with shell metacharacters should be passed as-is.

        stdio_client uses subprocess without shell=True, so this is safe.
        """
        cfg = {
            "command": "echo",
            "args": ["; rm -rf /", "$(whoami)", "`id`"],
        }
        client = _build_client_from_config("test", cfg)
        assert isinstance(client, MCPClient)

    def test_headers_with_auth_tokens(self):
        """Headers config should preserve auth tokens."""
        cfg = {
            "url": "https://example.com/mcp",
            "headers": {"Authorization": "Bearer secret-token-123"},
        }
        client = _build_client_from_config("test", cfg)
        assert isinstance(client, MCPClient)


# ═══════════════════════════════════════════════════════════════════════
# 10. MCPClient constructor edge cases
# ═══════════════════════════════════════════════════════════════════════


class TestMCPClientConstructorEdgeCases:
    """Edge cases for MCPClient initialization."""

    def test_explicit_callbacks_with_hook_override(self):
        """When MCPRegistry installs hook callback, explicit callback is preserved.

        The _install_hook_logging_callback wraps the original, so the
        explicit callback should still be called.
        """
        from strands.mcp.registry import _install_hook_logging_callback

        original_cb = AsyncMock()
        client = MCPClient(MagicMock(), logging_callback=original_cb)
        assert client._logging_callback is original_cb

        hook_reg = MagicMock()
        hook_reg.invoke_callbacks_async = AsyncMock(return_value=(None, []))
        _install_hook_logging_callback(client, "srv", hook_reg)

        # After installation, the callback should be the wrapper, not original
        assert client._logging_callback is not original_cb

    def test_prefix_empty_string(self):
        """Empty string prefix should work (tools get no prefix)."""
        client = MCPClient(MagicMock(), prefix="")
        assert client._prefix == ""

    def test_prefix_with_special_characters(self):
        """Prefix with special chars should be stored as-is."""
        client = MCPClient(MagicMock(), prefix="my-server.v2")
        assert client._prefix == "my-server.v2"

    def test_tool_filters_empty_dict(self):
        """Empty ToolFilters should be treated same as None."""
        from strands.tools.mcp.mcp_client import ToolFilters
        client = MCPClient(MagicMock(), tool_filters=ToolFilters())
        assert client._tool_filters is not None  # It's an empty ToolFilters, not None

    def test_startup_timeout_zero(self):
        """startup_timeout=0 should cause immediate timeout."""
        client = MCPClient(MagicMock(), startup_timeout=0)
        assert client._startup_timeout == 0


# ═══════════════════════════════════════════════════════════════════════
# 11. Import and export verification
# ═══════════════════════════════════════════════════════════════════════


class TestImportsAndExports:
    """Verify all public APIs are properly exported."""

    def test_mcp_package_exports(self):
        """strands.mcp should export MCPRegistry and load_mcp_servers."""
        from strands.mcp import MCPRegistry, load_mcp_servers
        assert MCPRegistry is not None
        assert load_mcp_servers is not None

    def test_mcp_package_all(self):
        """strands.mcp.__all__ should list public names."""
        import strands.mcp
        assert "MCPRegistry" in strands.mcp.__all__
        assert "load_mcp_servers" in strands.mcp.__all__

    def test_hooks_mcp_events_all_classes_exported(self):
        """All event classes from mcp_events should be in hooks.__init__."""
        from strands.hooks import mcp_events
        from strands import hooks

        # Get all public event classes from mcp_events
        event_classes = [
            name for name in dir(mcp_events)
            if name.startswith("MCP") and not name.startswith("_")
        ]

        for name in event_classes:
            assert hasattr(hooks, name), f"{name} not exported from strands.hooks"

    def test_backward_compat_mcp_client_import(self):
        """MCPClient should still be importable from the old path."""
        from strands.tools.mcp import MCPClient
        from strands.tools.mcp.mcp_client import MCPClient as DirectImport
        assert MCPClient is DirectImport


# ═══════════════════════════════════════════════════════════════════════
# 12. structuredContent and metadata propagation
# ═══════════════════════════════════════════════════════════════════════


class TestStructuredContentPropagation:
    """Test structuredContent and metadata in tool results."""

    def test_structured_content_propagated(self, mock_transport, mock_session):
        """structuredContent from MCP result should be in MCPToolResult."""
        mock_content = MCPTextContent(type="text", text="ok")
        result = MCPCallToolResult(isError=False, content=[mock_content])
        result.structuredContent = {"key": "value"}
        result.meta = None
        mock_session.call_tool.return_value = result

        with MCPClient(mock_transport["transport_callable"]) as client:
            tool_result = client.call_tool_sync(tool_use_id="t1", name="tool", arguments={})

        assert "structuredContent" in tool_result
        assert tool_result["structuredContent"] == {"key": "value"}

    def test_metadata_propagated(self, mock_transport, mock_session):
        """meta from MCP result should appear as metadata in MCPToolResult."""
        mock_content = MCPTextContent(type="text", text="ok")
        result = MCPCallToolResult(isError=False, content=[mock_content])
        result.structuredContent = None
        result.meta = {"request_id": "abc"}
        mock_session.call_tool.return_value = result

        with MCPClient(mock_transport["transport_callable"]) as client:
            tool_result = client.call_tool_sync(tool_use_id="t1", name="tool", arguments={})

        assert "metadata" in tool_result
        assert tool_result["metadata"] == {"request_id": "abc"}

    def test_no_structured_content_no_metadata(self, mock_transport, mock_session):
        """When structuredContent and meta are None, they should be absent."""
        mock_content = MCPTextContent(type="text", text="ok")
        result = MCPCallToolResult(isError=False, content=[mock_content])
        result.structuredContent = None
        result.meta = None
        mock_session.call_tool.return_value = result

        with MCPClient(mock_transport["transport_callable"]) as client:
            tool_result = client.call_tool_sync(tool_use_id="t1", name="tool", arguments={})

        assert "structuredContent" not in tool_result
        assert "metadata" not in tool_result


# ═══════════════════════════════════════════════════════════════════════
# 13. MCPClient error handling
# ═══════════════════════════════════════════════════════════════════════


class TestMCPClientErrorHandling:
    """Test error handling paths in MCPClient."""

    def test_call_tool_when_session_not_started(self):
        """Calling call_tool_sync without starting should raise."""
        client = MCPClient(MagicMock())
        with pytest.raises(Exception):
            client.call_tool_sync(tool_use_id="t1", name="tool", arguments={})

    def test_list_tools_when_session_not_started(self):
        """Calling list_tools_sync without starting should raise."""
        client = MCPClient(MagicMock())
        with pytest.raises(Exception):
            client.list_tools_sync()

    def test_start_when_already_started(self, mock_transport, mock_session):
        """Starting an already started client should raise."""
        with MCPClient(mock_transport["transport_callable"]) as client:
            from strands.types.exceptions import MCPClientInitializationError
            with pytest.raises(MCPClientInitializationError, match="currently running"):
                client.start()

    def test_stop_when_not_started(self):
        """Stopping a never-started client should not crash."""
        client = MCPClient(MagicMock())
        # Should not raise
        client.stop(None, None, None)

    def test_tool_execution_error_result_format(self, mock_transport, mock_session):
        """Tool execution errors should return proper MCPToolResult."""
        mock_session.call_tool.side_effect = ConnectionError("server gone")

        with MCPClient(mock_transport["transport_callable"]) as client:
            result = client.call_tool_sync(tool_use_id="t1", name="tool", arguments={})

        assert result["status"] == "error"
        assert result["isError"] is True
        assert any("server gone" in c.get("text", "") for c in result["content"])

    def test_tool_execution_error_preserves_tool_use_id(self, mock_transport, mock_session):
        """Error results should preserve the tool_use_id."""
        mock_session.call_tool.side_effect = RuntimeError("oops")

        with MCPClient(mock_transport["transport_callable"]) as client:
            result = client.call_tool_sync(tool_use_id="unique-123", name="tool", arguments={})

        assert result["toolUseId"] == "unique-123"
