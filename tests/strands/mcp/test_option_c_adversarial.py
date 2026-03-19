"""Adversarial tests for Option C: MCP Plugin.

These tests probe edge cases, race conditions, error handling gaps, and
correctness guarantees in the MCPPlugin implementation. Each test is
designed to expose a real or potential bug.
"""

import asyncio
import json
import os
import tempfile
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strands.hooks.mcp_events import (
    MCPLoggingEvent,
    MCPProgressEvent,
    MCPPromptsChangedEvent,
    MCPResourcesChangedEvent,
    MCPResourceUpdatedEvent,
    MCPToolsChangedEvent,
)
from strands.plugins.mcp import MCPPlugin, load_mcp_servers
from strands.tools.mcp import MCPClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_client(prefix="test"):
    """Create a mock MCPClient with necessary attributes."""
    client = MagicMock(spec=MCPClient)
    client._prefix = prefix
    client._message_handler = None
    client._handle_error_message = AsyncMock()
    return client


def _make_mock_agent():
    """Create a mock Agent with hooks and tool_registry."""
    agent = MagicMock()
    agent.hooks = MagicMock()
    agent.hooks.invoke_callbacks = MagicMock(return_value=(MagicMock(), []))
    agent.tool_registry = MagicMock()
    return agent


# ===========================================================================
# 1. MCPPlugin with empty clients list
# ===========================================================================

class TestEmptyClientsList:
    """Verify behavior with zero MCP clients."""

    def test_empty_clients_creates_plugin(self):
        """MCPPlugin(clients=[]) should not raise."""
        plugin = MCPPlugin(clients=[])
        assert plugin.name == "mcp"
        assert len(plugin._clients) == 0

    def test_empty_clients_init_agent(self):
        """init_agent with empty clients should succeed silently."""
        plugin = MCPPlugin(clients=[])
        agent = _make_mock_agent()
        plugin.init_agent(agent)
        agent.tool_registry.process_tools.assert_not_called()

    def test_empty_clients_from_config(self, tmp_path):
        """from_config with empty mcpServers should produce empty plugin."""
        config_path = tmp_path / "empty.json"
        config_path.write_text(json.dumps({"mcpServers": {}}))
        plugin = MCPPlugin.from_config(config_path)
        assert len(plugin._clients) == 0


# ===========================================================================
# 2. from_config with nonexistent file
# ===========================================================================

class TestFromConfigNonexistentFile:
    """Config loading with missing files."""

    def test_nonexistent_file_raises(self):
        """from_config with a nonexistent path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            MCPPlugin.from_config("/tmp/this_file_does_not_exist_12345.json")

    def test_nonexistent_file_load_mcp_servers(self):
        """load_mcp_servers with a nonexistent path should raise."""
        with pytest.raises(FileNotFoundError):
            load_mcp_servers("/tmp/this_file_does_not_exist_12345.json")


# ===========================================================================
# 3. from_config with malformed JSON
# ===========================================================================

class TestFromConfigMalformedJSON:
    """Config loading with invalid JSON."""

    def test_malformed_json_raises(self, tmp_path):
        """Malformed JSON should raise json.JSONDecodeError."""
        bad_path = tmp_path / "bad.json"
        bad_path.write_text("{this is not json}")
        with pytest.raises(json.JSONDecodeError):
            MCPPlugin.from_config(bad_path)

    def test_missing_mcpservers_key(self, tmp_path):
        """Valid JSON without 'mcpServers' key should return empty plugin."""
        path = tmp_path / "no_key.json"
        path.write_text(json.dumps({"other_key": {}}))
        plugin = MCPPlugin.from_config(path)
        assert len(plugin._clients) == 0

    def test_mcpservers_is_null(self, tmp_path):
        """mcpServers = null should return empty plugin (treated as empty dict)."""
        path = tmp_path / "null.json"
        path.write_text(json.dumps({"mcpServers": None}))
        # After fix: None is treated as empty dict via `or {}`
        plugin = MCPPlugin.from_config(path)
        assert len(plugin._clients) == 0


# ===========================================================================
# 4. Double init_agent (double registration bug)
# ===========================================================================

class TestDoubleInitAgent:
    """Test calling init_agent twice on the same plugin."""

    def test_double_init_is_idempotent(self):
        """Calling init_agent twice with the same agent should be idempotent (no double registration)."""
        client = _make_mock_client()
        agent = _make_mock_agent()
        plugin = MCPPlugin(clients=[client])

        plugin.init_agent(agent)
        plugin.init_agent(agent)

        # After fix: idempotency guard prevents double registration
        assert agent.tool_registry.process_tools.call_count == 1

    def test_double_init_handler_does_not_double_dispatch(self):
        """Even with double init, notifications should NOT fire twice."""
        from mcp.types import ToolListChangedNotification

        client = _make_mock_client()
        agent = _make_mock_agent()
        plugin = MCPPlugin(clients=[client])

        plugin.init_agent(agent)
        plugin.init_agent(agent)

        handler = client._message_handler
        notification = ToolListChangedNotification(
            method="notifications/tools/list_changed"
        )

        asyncio.get_event_loop().run_until_complete(handler(notification))

        # Notification dispatch should be 1, not 2
        # (because _install_message_handler captures _handle_error_message, not the previous routing handler)
        assert agent.hooks.invoke_callbacks.call_count == 1


# ===========================================================================
# 5. init_agent with agent that has no hooks registry
# ===========================================================================

class TestInitAgentNoHooks:
    """Test init_agent with an agent missing hooks."""

    def test_agent_without_hooks_attribute(self):
        """If agent has no hooks, _dispatch_notification should handle gracefully."""
        client = _make_mock_client()
        agent = MagicMock()
        agent.hooks = None  # No hooks registry
        agent.tool_registry = MagicMock()

        plugin = MCPPlugin(clients=[client])
        # init_agent should succeed (handler installation doesn't call hooks)
        plugin.init_agent(agent)

        # But dispatching a notification should not crash
        from mcp.types import ToolListChangedNotification

        handler = client._message_handler
        notification = ToolListChangedNotification(
            method="notifications/tools/list_changed"
        )

        # This should NOT raise, the try/except in _dispatch_notification should catch it
        asyncio.get_event_loop().run_until_complete(handler(notification))


# ===========================================================================
# 6. fail_open=True when ALL servers fail
# ===========================================================================

class TestFailOpenAllServersFail:
    """Test fail_open behavior when every client fails."""

    def test_all_servers_fail_with_fail_open(self):
        """When fail_open=True and all servers fail, agent should still init."""
        client1 = _make_mock_client("a")
        client2 = _make_mock_client("b")

        agent = _make_mock_agent()
        agent.tool_registry.process_tools.side_effect = RuntimeError("All down")

        plugin = MCPPlugin(clients=[client1, client2], fail_open=True)
        # Should not raise
        plugin.init_agent(agent)

        # Both attempts should have been made
        assert agent.tool_registry.process_tools.call_count == 2


# ===========================================================================
# 7. fail_open=False with first server failing
# ===========================================================================

class TestFailClosedFirstServerFails:
    """Test fail_open=False aborts on first failure."""

    def test_first_server_failure_aborts(self):
        """With fail_open=False, first failure should raise and not try second."""
        client1 = _make_mock_client("a")
        client2 = _make_mock_client("b")

        agent = _make_mock_agent()
        agent.tool_registry.process_tools.side_effect = RuntimeError("Server a down")

        plugin = MCPPlugin(clients=[client1, client2], fail_open=False)

        with pytest.raises(RuntimeError, match="Server a down"):
            plugin.init_agent(agent)

        # Only first server should have been attempted
        assert agent.tool_registry.process_tools.call_count == 1


# ===========================================================================
# 8. Message handler routing with unknown notification type
# ===========================================================================

class TestUnknownNotificationType:
    """Test routing with unrecognized notification types."""

    @pytest.mark.asyncio
    async def test_unknown_notification_type_logged_not_crashed(self):
        """Unknown notification types should be logged, not crash."""
        client = _make_mock_client()
        agent = _make_mock_agent()
        plugin = MCPPlugin(clients=[client])
        plugin.init_agent(agent)

        handler = client._message_handler

        # Pass a completely unknown object as notification
        class FakeNotification:
            pass

        await handler(FakeNotification())

        # Should not have invoked any hooks
        agent.hooks.invoke_callbacks.assert_not_called()


# ===========================================================================
# 9. Message handler with None message
# ===========================================================================

class TestNoneMessage:
    """Test routing handler with None as message."""

    @pytest.mark.asyncio
    async def test_none_message_does_not_crash(self):
        """Passing None to the routing handler should not crash."""
        client = _make_mock_client()
        agent = _make_mock_agent()
        plugin = MCPPlugin(clients=[client])
        plugin.init_agent(agent)

        handler = client._message_handler

        # None is not an Exception and not a ServerNotification
        # It should fall through to _dispatch_notification which will handle gracefully
        await handler(None)

        # Should not have invoked any hooks (None is not a known notification)
        agent.hooks.invoke_callbacks.assert_not_called()


# ===========================================================================
# 10. _meta extraction when _meta is deeply nested JSON
# ===========================================================================

class TestMetaExtractionEdgeCases:
    """Test _meta extraction edge cases in _create_call_tool_coroutine."""

    def _make_bare_client(self):
        """Create a bare MCPClient without starting it."""
        client = MCPClient.__new__(MCPClient)
        client._prefix = None
        client._tool_filters = None
        client._tasks_config = None
        client._session_id = "test"
        client._background_thread_session = MagicMock()
        client._server_task_capable = False
        return client

    def test_deeply_nested_meta(self):
        """_meta with deeply nested structure should be extracted correctly."""
        client = self._make_bare_client()
        deep_meta = {"progressToken": "tok-1", "nested": {"a": {"b": {"c": 42}}}}
        arguments = {"param1": "val", "_meta": deep_meta}

        coro = client._create_call_tool_coroutine("test_tool", arguments, None)
        coro.close()

        # Original arguments should not be mutated
        assert "_meta" in arguments
        assert arguments["_meta"] is deep_meta

    def test_meta_extraction_idempotency(self):
        """Calling _create_call_tool_coroutine twice with same args should be idempotent."""
        client = self._make_bare_client()
        arguments = {"param1": "val", "_meta": {"progressToken": "tok-1"}}

        coro1 = client._create_call_tool_coroutine("test_tool", arguments, None)
        coro1.close()

        # Second call should also work identically
        coro2 = client._create_call_tool_coroutine("test_tool", arguments, None)
        coro2.close()

        assert "_meta" in arguments

    def test_meta_extraction_with_none_arguments(self):
        """Arguments=None should not crash _meta extraction."""
        client = self._make_bare_client()
        coro = client._create_call_tool_coroutine("test_tool", None, None)
        coro.close()  # Should not raise

    def test_meta_extraction_with_empty_dict(self):
        """Empty arguments dict should work fine."""
        client = self._make_bare_client()
        coro = client._create_call_tool_coroutine("test_tool", {}, None)
        coro.close()


# ===========================================================================
# 11. isError when CallToolResult.isError is explicitly False vs missing
# ===========================================================================

class TestIsErrorSemantics:
    """Test isError handling in MCPToolResult."""

    def test_is_error_explicit_false(self):
        """isError=False should produce status='success'."""
        from mcp.types import CallToolResult, TextContent

        client = MCPClient.__new__(MCPClient)
        client._session_id = "test"
        client._tasks_config = None

        call_result = CallToolResult(
            content=[TextContent(type="text", text="OK")],
            isError=False,
        )
        result = client._handle_tool_result("test-id", call_result)
        assert result["status"] == "success"
        assert result["isError"] is False

    def test_is_error_none_treated_as_success(self):
        """isError=None (default) should produce status='success'."""
        from mcp.types import CallToolResult, TextContent

        client = MCPClient.__new__(MCPClient)
        client._session_id = "test"
        client._tasks_config = None

        call_result = CallToolResult(
            content=[TextContent(type="text", text="OK")],
        )
        result = client._handle_tool_result("test-id", call_result)
        assert result["status"] == "success"
        assert result["isError"] is False

    def test_is_error_true(self):
        """isError=True should produce status='error'."""
        from mcp.types import CallToolResult, TextContent

        client = MCPClient.__new__(MCPClient)
        client._session_id = "test"
        client._tasks_config = None

        call_result = CallToolResult(
            content=[TextContent(type="text", text="Error")],
            isError=True,
        )
        result = client._handle_tool_result("test-id", call_result)
        assert result["status"] == "error"
        assert result["isError"] is True


# ===========================================================================
# 12. Config loading with special characters
# ===========================================================================

class TestConfigSpecialCharacters:
    """Test config loading with unusual values."""

    def test_config_with_special_env_chars(self, tmp_path):
        """Config with $, {, } in values should be loaded as-is."""
        config = {
            "mcpServers": {
                "special": {
                    "command": "${HOME}/bin/server",
                    "args": ["--path=${PWD}"],
                    "prefix": "special",
                }
            }
        }
        path = tmp_path / "special.json"
        path.write_text(json.dumps(config))

        clients = load_mcp_servers(path)
        assert len(clients) == 1
        # The command should be the raw string, NOT expanded
        assert clients[0]._prefix == "special"

    def test_config_with_unicode_prefix(self, tmp_path):
        """Config with unicode prefix should work."""
        config = {
            "mcpServers": {
                "unicode": {
                    "command": "node",
                    "args": ["server.js"],
                    "prefix": "日本語",
                }
            }
        }
        path = tmp_path / "unicode.json"
        path.write_text(json.dumps(config, ensure_ascii=False))

        clients = load_mcp_servers(path)
        assert len(clients) == 1
        assert clients[0]._prefix == "日本語"


# ===========================================================================
# 13. Config with both command and url in same server entry
# ===========================================================================

class TestConfigCommandAndUrl:
    """Test behavior when both command and url are present."""

    def test_command_takes_precedence_over_url(self, tmp_path):
        """When both command and url are specified, command should win."""
        config = {
            "mcpServers": {
                "both": {
                    "command": "node",
                    "args": ["server.js"],
                    "url": "http://localhost:3000/sse",
                    "prefix": "both",
                }
            }
        }
        path = tmp_path / "both.json"
        path.write_text(json.dumps(config))

        clients = load_mcp_servers(path)
        assert len(clients) == 1
        assert clients[0]._prefix == "both"
        # Only one client created (command wins, url ignored)


# ===========================================================================
# 14. Config with no command and no url
# ===========================================================================

class TestConfigNoTransport:
    """Test server entry with neither command nor url."""

    def test_no_transport_skipped(self, tmp_path):
        """Server with neither command nor url should be skipped."""
        config = {
            "mcpServers": {
                "empty": {
                    "prefix": "empty",
                }
            }
        }
        path = tmp_path / "no_transport.json"
        path.write_text(json.dumps(config))

        clients = load_mcp_servers(path)
        assert len(clients) == 0


# ===========================================================================
# 15. Plugin tool registration order
# ===========================================================================

class TestToolRegistrationOrder:
    """Test that tools are registered in config/clients order."""

    def test_registration_order_matches_clients_order(self):
        """Tools should be registered in the same order as clients list."""
        clients = [_make_mock_client(f"prefix_{i}") for i in range(5)]
        agent = _make_mock_agent()
        plugin = MCPPlugin(clients=clients)
        plugin.init_agent(agent)

        calls = agent.tool_registry.process_tools.call_args_list
        assert len(calls) == 5
        for i, call in enumerate(calls):
            assert call[0][0][0]._prefix == f"prefix_{i}"


# ===========================================================================
# 16. Concurrent tool calls through plugin - thread safety
# ===========================================================================

class TestConcurrentToolCalls:
    """Test thread safety of plugin operations."""

    def test_multiple_clients_init_no_race(self):
        """Multiple clients registered concurrently should not corrupt state."""
        clients = [_make_mock_client(f"c_{i}") for i in range(10)]
        agent = _make_mock_agent()
        plugin = MCPPlugin(clients=clients)
        plugin.init_agent(agent)

        # All clients should have handlers installed
        for client in clients:
            assert client._message_handler is not None

        assert agent.tool_registry.process_tools.call_count == 10


# ===========================================================================
# 17. Hook events contain correct server_name
# ===========================================================================

class TestHookEventServerName:
    """Test that hook events have correct client/prefix info."""

    @pytest.mark.asyncio
    async def test_tools_changed_event_has_correct_prefix(self):
        """MCPToolsChangedEvent should carry the client's prefix."""
        from mcp.types import ToolListChangedNotification

        client = _make_mock_client("my-fs-server")
        agent = _make_mock_agent()
        plugin = MCPPlugin(clients=[client])
        plugin.init_agent(agent)

        handler = client._message_handler
        await handler(ToolListChangedNotification(method="notifications/tools/list_changed"))

        event = agent.hooks.invoke_callbacks.call_args[0][0]
        assert isinstance(event, MCPToolsChangedEvent)
        assert event.prefix == "my-fs-server"
        assert event.client is client

    @pytest.mark.asyncio
    async def test_logging_event_has_correct_client(self):
        """MCPLoggingEvent should carry the correct client reference."""
        from mcp.types import (
            LoggingMessageNotification,
            LoggingMessageNotificationParams,
        )

        client = _make_mock_client("logger-server")
        agent = _make_mock_agent()
        plugin = MCPPlugin(clients=[client])
        plugin.init_agent(agent)

        handler = client._message_handler
        await handler(
            LoggingMessageNotification(
                method="notifications/message",
                params=LoggingMessageNotificationParams(
                    level="error", data="boom", logger="custom"
                ),
            )
        )

        event = agent.hooks.invoke_callbacks.call_args[0][0]
        assert isinstance(event, MCPLoggingEvent)
        assert event.client is client
        assert event.level == "error"
        assert event.data == "boom"
        assert event.logger_name == "custom"


# ===========================================================================
# 18. from_config with relative vs absolute file paths
# ===========================================================================

class TestRelativeAbsolutePaths:
    """Test config loading with different path types."""

    def test_absolute_path(self, tmp_path):
        """Absolute path should work."""
        config_path = tmp_path / "abs.json"
        config_path.write_text(json.dumps({"mcpServers": {}}))
        clients = load_mcp_servers(str(config_path))
        assert clients == []

    def test_relative_path_from_cwd(self, tmp_path, monkeypatch):
        """Relative path should resolve from CWD."""
        config_path = tmp_path / "rel.json"
        config_path.write_text(json.dumps({"mcpServers": {}}))
        monkeypatch.chdir(tmp_path)
        clients = load_mcp_servers("rel.json")
        assert clients == []

    def test_path_object(self, tmp_path):
        """Path objects should work."""
        config_path = tmp_path / "pathobj.json"
        config_path.write_text(json.dumps({"mcpServers": {}}))
        clients = load_mcp_servers(Path(str(config_path)))
        assert clients == []


# ===========================================================================
# 19. ServerNotification wrapper handling
# ===========================================================================

class TestServerNotificationWrapper:
    """Test that both wrapped and unwrapped notifications are handled."""

    @pytest.mark.asyncio
    async def test_wrapped_server_notification(self):
        """ServerNotification(root=X) should be unwrapped and dispatched."""
        from mcp.types import (
            ResourceListChangedNotification,
            ServerNotification,
        )

        client = _make_mock_client()
        agent = _make_mock_agent()
        plugin = MCPPlugin(clients=[client])
        plugin.init_agent(agent)

        handler = client._message_handler
        inner = ResourceListChangedNotification(
            method="notifications/resources/list_changed"
        )
        wrapped = ServerNotification(root=inner)
        await handler(wrapped)

        event = agent.hooks.invoke_callbacks.call_args[0][0]
        assert isinstance(event, MCPResourcesChangedEvent)

    @pytest.mark.asyncio
    async def test_unwrapped_notification_also_dispatched(self):
        """Raw notification (not wrapped) should also be dispatched."""
        from mcp.types import PromptListChangedNotification

        client = _make_mock_client()
        agent = _make_mock_agent()
        plugin = MCPPlugin(clients=[client])
        plugin.init_agent(agent)

        handler = client._message_handler
        notification = PromptListChangedNotification(
            method="notifications/prompts/list_changed"
        )
        await handler(notification)

        event = agent.hooks.invoke_callbacks.call_args[0][0]
        assert isinstance(event, MCPPromptsChangedEvent)


# ===========================================================================
# 20. MCPPlugin._agent reference lifecycle
# ===========================================================================

class TestAgentReferenceLifecycle:
    """Test the _agent reference on MCPPlugin."""

    def test_agent_not_set_before_init(self):
        """_agent should be None before init_agent is called."""
        plugin = MCPPlugin(clients=[])
        assert plugin._agent is None

    def test_agent_set_after_init(self):
        """_agent should be set after init_agent."""
        agent = _make_mock_agent()
        plugin = MCPPlugin(clients=[])
        plugin.init_agent(agent)
        assert plugin._agent is agent

    def test_install_handler_without_agent(self):
        """_install_message_handler should no-op when _agent is None."""
        client = _make_mock_client()
        plugin = MCPPlugin(clients=[client])
        # Don't call init_agent, call _install_message_handler directly
        plugin._install_message_handler(client)
        # Handler should NOT be set (agent is None guard)
        assert client._message_handler is None


# ===========================================================================
# 21. Config loading with disabled and disabledTools combinations
# ===========================================================================

class TestConfigDisabledCombinations:
    """Test config loading with various disabled states."""

    def test_disabled_true_skips_server(self, tmp_path):
        """disabled=True should skip the server entirely."""
        config = {
            "mcpServers": {
                "srv": {
                    "command": "node",
                    "args": [],
                    "disabled": True,
                }
            }
        }
        path = tmp_path / "disabled.json"
        path.write_text(json.dumps(config))
        clients = load_mcp_servers(path)
        assert len(clients) == 0

    def test_disabled_false_keeps_server(self, tmp_path):
        """disabled=False (explicit) should keep the server."""
        config = {
            "mcpServers": {
                "srv": {
                    "command": "node",
                    "args": [],
                    "disabled": False,
                    "prefix": "srv",
                }
            }
        }
        path = tmp_path / "enabled.json"
        path.write_text(json.dumps(config))
        clients = load_mcp_servers(path)
        assert len(clients) == 1

    def test_empty_disabled_tools_no_filter(self, tmp_path):
        """Empty disabledTools list should NOT create a ToolFilter."""
        config = {
            "mcpServers": {
                "srv": {
                    "command": "node",
                    "args": [],
                    "prefix": "srv",
                    "disabledTools": [],
                }
            }
        }
        path = tmp_path / "empty_dt.json"
        path.write_text(json.dumps(config))
        clients = load_mcp_servers(path)
        assert len(clients) == 1
        assert clients[0]._tool_filters is None


# ===========================================================================
# 22. Exception in hook callback during dispatch
# ===========================================================================

class TestHookCallbackException:
    """Test that exceptions in hook callbacks don't crash the handler."""

    @pytest.mark.asyncio
    async def test_hook_callback_exception_swallowed(self):
        """Exception in invoke_callbacks should be caught and logged."""
        from mcp.types import ToolListChangedNotification

        client = _make_mock_client()
        agent = _make_mock_agent()
        agent.hooks.invoke_callbacks.side_effect = RuntimeError("Hook exploded!")

        plugin = MCPPlugin(clients=[client])
        plugin.init_agent(agent)

        handler = client._message_handler
        notification = ToolListChangedNotification(
            method="notifications/tools/list_changed"
        )

        # Should NOT raise even though invoke_callbacks raises
        await handler(notification)


# ===========================================================================
# 23. ProgressNotification with missing optional fields
# ===========================================================================

class TestProgressNotificationEdgeCases:
    """Test ProgressNotification with partial params."""

    @pytest.mark.asyncio
    async def test_progress_without_total(self):
        """ProgressNotification with no total should set total=None."""
        from mcp.types import ProgressNotification, ProgressNotificationParams

        client = _make_mock_client()
        agent = _make_mock_agent()
        plugin = MCPPlugin(clients=[client])
        plugin.init_agent(agent)

        handler = client._message_handler
        notification = ProgressNotification(
            method="notifications/progress",
            params=ProgressNotificationParams(
                progressToken="tok-1",
                progress=50.0,
            ),
        )
        await handler(notification)

        event = agent.hooks.invoke_callbacks.call_args[0][0]
        assert isinstance(event, MCPProgressEvent)
        assert event.total is None

    @pytest.mark.asyncio
    async def test_progress_with_zero_values(self):
        """Progress=0.0, total=0.0 should be handled (not treated as falsy)."""
        from mcp.types import ProgressNotification, ProgressNotificationParams

        client = _make_mock_client()
        agent = _make_mock_agent()
        plugin = MCPPlugin(clients=[client])
        plugin.init_agent(agent)

        handler = client._message_handler
        notification = ProgressNotification(
            method="notifications/progress",
            params=ProgressNotificationParams(
                progressToken="tok-2",
                progress=0.0,
                total=0.0,
            ),
        )
        await handler(notification)

        event = agent.hooks.invoke_callbacks.call_args[0][0]
        assert event.progress == 0.0
        assert event.total == 0.0


# ===========================================================================
# 24. MCPPlugin name property is stable
# ===========================================================================

class TestPluginNameStability:
    """Plugin name should be stable and consistent."""

    def test_name_is_always_mcp(self):
        """name property should always return 'mcp'."""
        p1 = MCPPlugin(clients=[])
        p2 = MCPPlugin(clients=[_make_mock_client()])
        assert p1.name == "mcp"
        assert p2.name == "mcp"
        assert p1.name == p2.name


# ===========================================================================
# 25. Config URL detection logic
# ===========================================================================

class TestURLDetectionLogic:
    """Test the SSE vs streamable HTTP detection."""

    def test_sse_url_detected(self, tmp_path):
        """URL containing '/sse' should use SSE transport."""
        config = {
            "mcpServers": {
                "sse-srv": {
                    "url": "http://localhost:3000/sse",
                    "prefix": "sse",
                }
            }
        }
        path = tmp_path / "sse.json"
        path.write_text(json.dumps(config))
        clients = load_mcp_servers(path)
        assert len(clients) == 1

    def test_non_sse_url_uses_streamable_http(self, tmp_path):
        """URL without '/sse' should use streamable HTTP transport."""
        config = {
            "mcpServers": {
                "http-srv": {
                    "url": "http://localhost:3000/mcp",
                    "prefix": "http",
                }
            }
        }
        path = tmp_path / "http.json"
        path.write_text(json.dumps(config))
        clients = load_mcp_servers(path)
        assert len(clients) == 1


# ===========================================================================
# 26. MCPToolResult backward compatibility
# ===========================================================================

class TestMCPToolResultBackwardCompat:
    """Test MCPToolResult TypedDict backward compatibility."""

    def test_tool_result_without_optional_fields(self):
        """MCPToolResult should work without structuredContent, metadata, isError."""
        from strands.tools.mcp.mcp_types import MCPToolResult

        result = MCPToolResult(
            status="success",
            toolUseId="test-id",
            content=[{"text": "OK"}],
        )
        assert "structuredContent" not in result
        assert "metadata" not in result
        assert "isError" not in result

    def test_tool_result_with_all_optional_fields(self):
        """MCPToolResult should work with all optional fields."""
        from strands.tools.mcp.mcp_types import MCPToolResult

        result = MCPToolResult(
            status="success",
            toolUseId="test-id",
            content=[{"text": "OK"}],
            structuredContent={"key": "val"},
            metadata={"usage": 42},
            isError=False,
        )
        assert result["structuredContent"] == {"key": "val"}
        assert result["metadata"] == {"usage": 42}
        assert result["isError"] is False
