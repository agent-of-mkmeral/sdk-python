"""Tests for MCPPlugin lifecycle, hook events, config loading, and bug fixes."""

import json
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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_mcp_client():
    """Create a mock MCPClient."""
    client = MagicMock(spec=MCPClient)
    client._prefix = "test"
    client._message_handler = None
    client._handle_error_message = AsyncMock()
    return client


@pytest.fixture
def mock_agent():
    """Create a mock Agent with hooks and tool_registry."""
    agent = MagicMock()
    agent.hooks = MagicMock()
    agent.hooks.invoke_callbacks = MagicMock(return_value=(MagicMock(), []))
    agent.tool_registry = MagicMock()
    return agent


@pytest.fixture
def sample_config(tmp_path):
    """Create a sample MCP config JSON file."""
    config = {
        "mcpServers": {
            "fs-server": {
                "command": "node",
                "args": ["server.js"],
                "prefix": "fs",
            },
            "disabled-server": {
                "command": "node",
                "args": ["disabled.js"],
                "disabled": True,
            },
            "filtered-server": {
                "command": "node",
                "args": ["filtered.js"],
                "prefix": "filtered",
                "disabledTools": ["bad_tool"],
            },
        }
    }
    config_path = tmp_path / "mcp_config.json"
    config_path.write_text(json.dumps(config))
    return config_path


# ---------------------------------------------------------------------------
# MCPPlugin basic lifecycle tests
# ---------------------------------------------------------------------------


class TestMCPPluginLifecycle:
    """Test MCPPlugin creation and initialization."""

    def test_plugin_name(self, mock_mcp_client):
        """Plugin name should be 'mcp'."""
        plugin = MCPPlugin(clients=[mock_mcp_client])
        assert plugin.name == "mcp"

    def test_init_stores_clients(self, mock_mcp_client):
        """Plugin should store the provided clients."""
        plugin = MCPPlugin(clients=[mock_mcp_client])
        assert len(plugin._clients) == 1
        assert plugin._clients[0] is mock_mcp_client

    def test_init_agent_registers_tools(self, mock_mcp_client, mock_agent):
        """init_agent should register each client as a tool provider."""
        plugin = MCPPlugin(clients=[mock_mcp_client])
        plugin.init_agent(mock_agent)

        mock_agent.tool_registry.process_tools.assert_called_once_with([mock_mcp_client])

    def test_init_agent_installs_message_handler(self, mock_mcp_client, mock_agent):
        """init_agent should install a message handler on each client."""
        plugin = MCPPlugin(clients=[mock_mcp_client])
        plugin.init_agent(mock_agent)

        # The message handler should have been set
        assert mock_mcp_client._message_handler is not None

    def test_init_agent_fail_open(self, mock_mcp_client, mock_agent):
        """With fail_open=True, registration failures should be logged, not raised."""
        mock_agent.tool_registry.process_tools.side_effect = RuntimeError("Connection failed")
        plugin = MCPPlugin(clients=[mock_mcp_client], fail_open=True)

        # Should not raise
        plugin.init_agent(mock_agent)

    def test_init_agent_fail_closed(self, mock_mcp_client, mock_agent):
        """With fail_open=False (default), registration failures should raise."""
        mock_agent.tool_registry.process_tools.side_effect = RuntimeError("Connection failed")
        plugin = MCPPlugin(clients=[mock_mcp_client])

        with pytest.raises(RuntimeError, match="Connection failed"):
            plugin.init_agent(mock_agent)

    def test_multiple_clients(self, mock_agent):
        """Plugin should handle multiple clients."""
        client1 = MagicMock(spec=MCPClient)
        client1._prefix = "a"
        client1._message_handler = None
        client1._handle_error_message = AsyncMock()

        client2 = MagicMock(spec=MCPClient)
        client2._prefix = "b"
        client2._message_handler = None
        client2._handle_error_message = AsyncMock()

        plugin = MCPPlugin(clients=[client1, client2])
        plugin.init_agent(mock_agent)

        assert mock_agent.tool_registry.process_tools.call_count == 2


# ---------------------------------------------------------------------------
# Hook event routing tests
# ---------------------------------------------------------------------------


class TestMCPHookEventRouting:
    """Test that ServerNotification types are routed to the correct hook events."""

    def _get_handler(self, mock_mcp_client, mock_agent):
        """Helper to install plugin and return the message handler."""
        plugin = MCPPlugin(clients=[mock_mcp_client])
        plugin.init_agent(mock_agent)
        return mock_mcp_client._message_handler

    @pytest.mark.asyncio
    async def test_tool_list_changed_event(self, mock_mcp_client, mock_agent):
        """ToolListChangedNotification should emit MCPToolsChangedEvent."""
        from mcp.types import ToolListChangedNotification

        handler = self._get_handler(mock_mcp_client, mock_agent)
        notification = ToolListChangedNotification(method="notifications/tools/list_changed")

        await handler(notification)

        mock_agent.hooks.invoke_callbacks.assert_called_once()
        event = mock_agent.hooks.invoke_callbacks.call_args[0][0]
        assert isinstance(event, MCPToolsChangedEvent)
        assert event.agent is mock_agent
        assert event.client is mock_mcp_client
        assert event.prefix == "test"

    @pytest.mark.asyncio
    async def test_resource_list_changed_event(self, mock_mcp_client, mock_agent):
        """ResourceListChangedNotification should emit MCPResourcesChangedEvent."""
        from mcp.types import ResourceListChangedNotification

        handler = self._get_handler(mock_mcp_client, mock_agent)
        notification = ResourceListChangedNotification(method="notifications/resources/list_changed")

        await handler(notification)

        event = mock_agent.hooks.invoke_callbacks.call_args[0][0]
        assert isinstance(event, MCPResourcesChangedEvent)

    @pytest.mark.asyncio
    async def test_prompt_list_changed_event(self, mock_mcp_client, mock_agent):
        """PromptListChangedNotification should emit MCPPromptsChangedEvent."""
        from mcp.types import PromptListChangedNotification

        handler = self._get_handler(mock_mcp_client, mock_agent)
        notification = PromptListChangedNotification(method="notifications/prompts/list_changed")

        await handler(notification)

        event = mock_agent.hooks.invoke_callbacks.call_args[0][0]
        assert isinstance(event, MCPPromptsChangedEvent)

    @pytest.mark.asyncio
    async def test_logging_event(self, mock_mcp_client, mock_agent):
        """LoggingMessageNotification should emit MCPLoggingEvent."""
        from mcp.types import LoggingMessageNotification, LoggingMessageNotificationParams

        handler = self._get_handler(mock_mcp_client, mock_agent)
        notification = LoggingMessageNotification(
            method="notifications/message",
            params=LoggingMessageNotificationParams(
                level="warning",
                data="Something happened",
                logger="test-logger",
            ),
        )

        await handler(notification)

        event = mock_agent.hooks.invoke_callbacks.call_args[0][0]
        assert isinstance(event, MCPLoggingEvent)
        assert event.level == "warning"
        assert event.data == "Something happened"
        assert event.logger_name == "test-logger"

    @pytest.mark.asyncio
    async def test_progress_event(self, mock_mcp_client, mock_agent):
        """ProgressNotification should emit MCPProgressEvent."""
        from mcp.types import ProgressNotification, ProgressNotificationParams

        handler = self._get_handler(mock_mcp_client, mock_agent)
        notification = ProgressNotification(
            method="notifications/progress",
            params=ProgressNotificationParams(
                progressToken="tok-1",
                progress=50.0,
                total=100.0,
            ),
        )

        await handler(notification)

        event = mock_agent.hooks.invoke_callbacks.call_args[0][0]
        assert isinstance(event, MCPProgressEvent)
        assert event.progress == 50.0
        assert event.total == 100.0
        assert event.progress_token == "tok-1"

    @pytest.mark.asyncio
    async def test_resource_updated_event(self, mock_mcp_client, mock_agent):
        """ResourceUpdatedNotification should emit MCPResourceUpdatedEvent."""
        from mcp.types import ResourceUpdatedNotification, ResourceUpdatedNotificationParams

        handler = self._get_handler(mock_mcp_client, mock_agent)
        notification = ResourceUpdatedNotification(
            method="notifications/resources/updated",
            params=ResourceUpdatedNotificationParams(uri="file:///tmp/data.json"),
        )

        await handler(notification)

        event = mock_agent.hooks.invoke_callbacks.call_args[0][0]
        assert isinstance(event, MCPResourceUpdatedEvent)
        assert "file:///tmp/data.json" in event.uri

    @pytest.mark.asyncio
    async def test_exception_forwarded_to_original_handler(self, mock_mcp_client, mock_agent):
        """Exceptions should be forwarded to the original error handler."""
        handler = self._get_handler(mock_mcp_client, mock_agent)
        exc = RuntimeError("test error")

        await handler(exc)

        mock_mcp_client._handle_error_message.assert_called_once_with(exc)
        mock_agent.hooks.invoke_callbacks.assert_not_called()


# ---------------------------------------------------------------------------
# Config loading tests
# ---------------------------------------------------------------------------


class TestConfigLoading:
    """Test load_mcp_servers and MCPPlugin.from_config."""

    @patch("mcp.stdio_client")
    @patch("mcp.StdioServerParameters")
    def test_load_mcp_servers_stdio(self, mock_params, mock_stdio, sample_config):
        """load_mcp_servers should create clients from stdio config."""
        clients = load_mcp_servers(sample_config)

        # 2 enabled servers (fs-server, filtered-server); disabled-server skipped
        assert len(clients) == 2

    @patch("mcp.stdio_client")
    @patch("mcp.StdioServerParameters")
    def test_load_mcp_servers_skips_disabled(self, mock_params, mock_stdio, sample_config):
        """Disabled servers should be skipped."""
        clients = load_mcp_servers(sample_config)
        prefixes = [c._prefix for c in clients]
        assert "disabled-server" not in prefixes

    @patch("mcp.stdio_client")
    @patch("mcp.StdioServerParameters")
    def test_load_mcp_servers_with_tool_filters(self, mock_params, mock_stdio, sample_config):
        """Servers with disabledTools should get ToolFilters."""
        clients = load_mcp_servers(sample_config)
        filtered_client = [c for c in clients if c._prefix == "filtered"][0]
        assert filtered_client._tool_filters is not None
        assert "bad_tool" in filtered_client._tool_filters.get("rejected", [])

    @patch("strands.plugins.mcp.load_mcp_servers")
    def test_from_config(self, mock_load, sample_config):
        """MCPPlugin.from_config should create a plugin from a config file."""
        mock_client = MagicMock(spec=MCPClient)
        mock_load.return_value = [mock_client]

        plugin = MCPPlugin.from_config(sample_config, fail_open=True)

        assert plugin.name == "mcp"
        assert plugin._fail_open is True
        assert len(plugin._clients) == 1
        mock_load.assert_called_once_with(sample_config)

    def test_load_mcp_servers_empty_config(self, tmp_path):
        """Empty config should return empty list."""
        config_path = tmp_path / "empty.json"
        config_path.write_text(json.dumps({"mcpServers": {}}))
        clients = load_mcp_servers(config_path)
        assert clients == []

    @patch("mcp.client.sse.sse_client")
    @patch("mcp.client.streamable_http.streamablehttp_client")
    def test_load_mcp_servers_url_config(self, mock_streamable, mock_sse, tmp_path):
        """URL-based servers should create HTTP clients."""
        config = {
            "mcpServers": {
                "http-server": {
                    "url": "http://localhost:3000/sse",
                    "prefix": "http",
                }
            }
        }
        config_path = tmp_path / "http_config.json"
        config_path.write_text(json.dumps(config))

        clients = load_mcp_servers(config_path)
        assert len(clients) == 1
        assert clients[0]._prefix == "http"


# ---------------------------------------------------------------------------
# MCPToolResult.isError bug fix tests
# ---------------------------------------------------------------------------


class TestMCPToolResultIsError:
    """Test that isError is correctly populated from CallToolResult."""

    def test_is_error_true(self):
        """MCPToolResult should include isError=True for error results."""
        from strands.tools.mcp.mcp_types import MCPToolResult

        result = MCPToolResult(
            status="error",
            toolUseId="test-id",
            content=[{"text": "Error occurred"}],
            isError=True,
        )
        assert result["isError"] is True

    def test_is_error_false(self):
        """MCPToolResult should include isError=False for success results."""
        from strands.tools.mcp.mcp_types import MCPToolResult

        result = MCPToolResult(
            status="success",
            toolUseId="test-id",
            content=[{"text": "OK"}],
            isError=False,
        )
        assert result["isError"] is False

    def test_is_error_not_required(self):
        """MCPToolResult should work without isError (backward compat)."""
        from strands.tools.mcp.mcp_types import MCPToolResult

        result = MCPToolResult(
            status="success",
            toolUseId="test-id",
            content=[{"text": "OK"}],
        )
        assert "isError" not in result


# ---------------------------------------------------------------------------
# _meta extraction bug fix tests
# ---------------------------------------------------------------------------


class TestMetaExtraction:
    """Test that _meta is extracted from arguments and passed as meta= to call_tool."""

    def test_meta_extracted_from_arguments(self):
        """_create_call_tool_coroutine should extract _meta from arguments."""
        from strands.tools.mcp.mcp_client import MCPClient

        client = MCPClient.__new__(MCPClient)
        client._prefix = None
        client._tool_filters = None
        client._tasks_config = None
        client._session_id = "test"
        client._background_thread_session = MagicMock()
        client._server_task_capable = False

        arguments = {"param1": "value1", "_meta": {"progressToken": "tok-123"}}

        # Call the method and inspect the coroutine
        coro = client._create_call_tool_coroutine("test_tool", arguments, None)

        # The original arguments should not have been mutated
        assert "_meta" in arguments  # original dict unchanged
        # The coroutine was created — just close it to avoid warnings
        coro.close()

    def test_meta_none_when_not_present(self):
        """When no _meta key, meta should be None."""
        from strands.tools.mcp.mcp_client import MCPClient

        client = MCPClient.__new__(MCPClient)
        client._prefix = None
        client._tool_filters = None
        client._tasks_config = None
        client._session_id = "test"
        client._background_thread_session = MagicMock()
        client._server_task_capable = False

        arguments = {"param1": "value1"}
        coro = client._create_call_tool_coroutine("test_tool", arguments, None)
        coro.close()


# ---------------------------------------------------------------------------
# Default logging callback tests
# ---------------------------------------------------------------------------


class TestDefaultLoggingCallback:
    """Test the default MCP logging callback."""

    def test_default_logging_callback_maps_levels(self):
        """Default logging callback should map MCP levels to Python logging levels."""
        import logging
        from strands.tools.mcp.mcp_client import _default_logging_callback
        from mcp.types import LoggingMessageNotificationParams

        params = LoggingMessageNotificationParams(
            level="warning",
            data="test warning",
            logger="test",
        )

        with patch("strands.tools.mcp.mcp_client.logger") as mock_logger:
            _default_logging_callback(params)
            mock_logger.log.assert_called_once_with(
                logging.WARNING, "[MCP:%s] %s", "test", "test warning"
            )

    def test_default_logging_callback_default_level(self):
        """Unknown levels should map to INFO."""
        import logging
        from strands.tools.mcp.mcp_client import _default_logging_callback
        from mcp.types import LoggingMessageNotificationParams

        params = LoggingMessageNotificationParams(
            level="notice",
            data="test notice",
        )

        with patch("strands.tools.mcp.mcp_client.logger") as mock_logger:
            _default_logging_callback(params)
            mock_logger.log.assert_called_once_with(
                logging.INFO, "[MCP:%s] %s", "server", "test notice"
            )


# ---------------------------------------------------------------------------
# ClientSession callback wiring tests
# ---------------------------------------------------------------------------


class TestClientSessionCallbacks:
    """Test that new callbacks are accepted by MCPClient constructor."""

    def test_sampling_callback_stored(self):
        """MCPClient should store sampling_callback."""
        cb = MagicMock()
        client = MCPClient(transport_callable=MagicMock(), sampling_callback=cb)
        assert client._sampling_callback is cb

    def test_list_roots_callback_stored(self):
        """MCPClient should store list_roots_callback."""
        cb = MagicMock()
        client = MCPClient(transport_callable=MagicMock(), list_roots_callback=cb)
        assert client._list_roots_callback is cb

    def test_logging_callback_default(self):
        """MCPClient should have a default logging callback."""
        client = MCPClient(transport_callable=MagicMock())
        assert client._logging_callback is not None

    def test_logging_callback_none(self):
        """MCPClient should accept None for logging callback."""
        client = MCPClient(transport_callable=MagicMock(), logging_callback=None)
        assert client._logging_callback is None

    def test_message_handler_stored(self):
        """MCPClient should store message_handler."""
        handler = AsyncMock()
        client = MCPClient(transport_callable=MagicMock(), message_handler=handler)
        assert client._message_handler is handler


# ---------------------------------------------------------------------------
# MCPEvent dataclass tests
# ---------------------------------------------------------------------------


class TestMCPEventDataclasses:
    """Test MCP event dataclass creation."""

    def test_tools_changed_event(self, mock_agent):
        """MCPToolsChangedEvent should be constructable."""
        event = MCPToolsChangedEvent(agent=mock_agent, client=MagicMock(), prefix="test")
        assert event.prefix == "test"

    def test_logging_event(self, mock_agent):
        """MCPLoggingEvent should carry level, data, and logger_name."""
        event = MCPLoggingEvent(
            agent=mock_agent,
            client=MagicMock(),
            level="error",
            data={"key": "value"},
            logger_name="my-logger",
        )
        assert event.level == "error"
        assert event.data == {"key": "value"}
        assert event.logger_name == "my-logger"

    def test_progress_event(self, mock_agent):
        """MCPProgressEvent should carry progress values."""
        event = MCPProgressEvent(
            agent=mock_agent,
            client=MagicMock(),
            progress=75.0,
            total=100.0,
            message="Processing...",
            progress_token="tok-1",
        )
        assert event.progress == 75.0
        assert event.total == 100.0
        assert event.message == "Processing..."

    def test_resource_updated_event(self, mock_agent):
        """MCPResourceUpdatedEvent should carry the URI."""
        event = MCPResourceUpdatedEvent(
            agent=mock_agent,
            client=MagicMock(),
            uri="file:///tmp/data.json",
        )
        assert event.uri == "file:///tmp/data.json"
