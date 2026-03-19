"""Tests for MCP client bug fixes: _meta extraction, isError propagation, and callback wiring."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import CallToolResult as MCPCallToolResult
from mcp.types import TextContent as MCPTextContent

from strands.tools.mcp import MCPClient
from strands.tools.mcp.mcp_client import _default_logging_callback
from strands.tools.mcp.mcp_types import MCPToolResult


# ── Fixtures (reuse conftest from tests/strands/tools/mcp) ──────────


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


# ── _meta extraction tests ───────────────────────────────────────────


class TestMetaExtraction:
    """call_tool should extract _meta from arguments and pass it as meta= kwarg."""

    def test_meta_extracted_and_passed(self, mock_transport, mock_session):
        mock_content = MCPTextContent(type="text", text="ok")
        mock_session.call_tool.return_value = MCPCallToolResult(isError=False, content=[mock_content])

        with MCPClient(mock_transport["transport_callable"]) as client:
            result = client.call_tool_sync(
                tool_use_id="t1",
                name="my_tool",
                arguments={"key": "val", "_meta": {"progressToken": "abc"}},
            )

        # call_tool should receive arguments WITHOUT _meta, and meta= as kwarg
        call_args = mock_session.call_tool.call_args
        assert call_args[0][0] == "my_tool"  # name
        assert call_args[0][1] == {"key": "val"}  # arguments (no _meta)
        assert call_args[1]["meta"] == {"progressToken": "abc"}

    def test_no_meta_passes_none(self, mock_transport, mock_session):
        mock_content = MCPTextContent(type="text", text="ok")
        mock_session.call_tool.return_value = MCPCallToolResult(isError=False, content=[mock_content])

        with MCPClient(mock_transport["transport_callable"]) as client:
            result = client.call_tool_sync(
                tool_use_id="t1",
                name="my_tool",
                arguments={"key": "val"},
            )

        call_args = mock_session.call_tool.call_args
        assert call_args[0][1] == {"key": "val"}
        assert call_args[1]["meta"] is None

    def test_meta_does_not_mutate_original(self, mock_transport, mock_session):
        """Ensure original arguments dict is not mutated by _meta extraction."""
        mock_content = MCPTextContent(type="text", text="ok")
        mock_session.call_tool.return_value = MCPCallToolResult(isError=False, content=[mock_content])

        original_args = {"key": "val", "_meta": {"progressToken": "abc"}}

        with MCPClient(mock_transport["transport_callable"]) as client:
            client.call_tool_sync(tool_use_id="t1", name="my_tool", arguments=original_args)

        # Original dict should still have _meta
        assert "_meta" in original_args


# ── isError propagation tests ────────────────────────────────────────


class TestIsErrorPropagation:
    """MCPToolResult should contain an isError field mirroring CallToolResult.isError."""

    @pytest.mark.parametrize("mcp_is_error,expected_status,expected_is_error", [
        (False, "success", False),
        (True, "error", True),
    ])
    def test_is_error_propagated(self, mock_transport, mock_session, mcp_is_error, expected_status, expected_is_error):
        mock_content = MCPTextContent(type="text", text="result")
        mock_session.call_tool.return_value = MCPCallToolResult(isError=mcp_is_error, content=[mock_content])

        with MCPClient(mock_transport["transport_callable"]) as client:
            result = client.call_tool_sync(tool_use_id="t1", name="tool", arguments={})

        assert result["status"] == expected_status
        assert result["isError"] == expected_is_error

    def test_error_handler_sets_is_error(self, mock_transport, mock_session):
        """When an exception occurs, isError should be True."""
        mock_session.call_tool.side_effect = RuntimeError("boom")

        with MCPClient(mock_transport["transport_callable"]) as client:
            result = client.call_tool_sync(tool_use_id="t1", name="tool", arguments={})

        assert result["status"] == "error"
        assert result["isError"] is True


# ── ClientSession callback wiring tests ──────────────────────────────


class TestCallbackWiring:
    """Verify new callback params are stored and passed to ClientSession."""

    def test_default_logging_callback(self):
        """Default logging callback should be the module-level function."""
        transport = MagicMock()
        client = MCPClient(transport)
        assert client._logging_callback is _default_logging_callback

    def test_custom_callbacks_stored(self):
        """Custom callbacks passed to __init__ should be stored."""
        transport = MagicMock()
        sampling_cb = MagicMock()
        roots_cb = MagicMock()
        logging_cb = MagicMock()
        progress_cb = MagicMock()

        client = MCPClient(
            transport,
            sampling_callback=sampling_cb,
            list_roots_callback=roots_cb,
            logging_callback=logging_cb,
            progress_callback=progress_cb,
        )

        assert client._sampling_callback is sampling_cb
        assert client._list_roots_callback is roots_cb
        assert client._logging_callback is logging_cb
        assert client._progress_callback is progress_cb

    def test_none_logging_callback(self):
        """Setting logging_callback=None should disable it."""
        transport = MagicMock()
        client = MCPClient(transport, logging_callback=None)
        assert client._logging_callback is None

    def test_callbacks_passed_to_client_session(self, mock_transport):
        """Verify that callbacks are passed through to ClientSession constructor."""
        sampling_cb = AsyncMock()
        roots_cb = AsyncMock()
        logging_cb = AsyncMock()

        with patch("strands.tools.mcp.mcp_client.ClientSession") as MockSession:
            mock_session = AsyncMock()
            mock_init_result = MagicMock()
            mock_init_result.instructions = None
            mock_session.initialize = AsyncMock(return_value=mock_init_result)
            mock_session.get_server_capabilities = MagicMock(return_value=None)
            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__.return_value = mock_session
            MockSession.return_value = mock_session_cm

            client = MCPClient(
                mock_transport["transport_callable"],
                sampling_callback=sampling_cb,
                list_roots_callback=roots_cb,
                logging_callback=logging_cb,
            )

            try:
                client.start()
                time.sleep(0.2)  # Give background thread time to run

                # Verify ClientSession was called with our callbacks
                session_kwargs = MockSession.call_args
                assert session_kwargs[1].get("sampling_callback") is sampling_cb
                assert session_kwargs[1].get("list_roots_callback") is roots_cb
                assert session_kwargs[1].get("logging_callback") is logging_cb
            finally:
                client.stop(None, None, None)

    def test_progress_callback_passed_to_call_tool(self, mock_transport, mock_session):
        """Verify progress_callback from __init__ is passed to session.call_tool."""
        progress_cb = MagicMock()
        mock_content = MCPTextContent(type="text", text="ok")
        mock_session.call_tool.return_value = MCPCallToolResult(isError=False, content=[mock_content])

        with MCPClient(mock_transport["transport_callable"], progress_callback=progress_cb) as client:
            client.call_tool_sync(tool_use_id="t1", name="tool", arguments={})

        call_kwargs = mock_session.call_tool.call_args[1]
        assert call_kwargs.get("progress_callback") is progress_cb


# ── Default logging callback tests ───────────────────────────────────


class TestDefaultLoggingCallback:
    """Tests for the _default_logging_callback function."""

    @pytest.mark.asyncio
    async def test_routes_to_python_logging(self):
        from mcp.types import LoggingMessageNotificationParams

        params = LoggingMessageNotificationParams(
            level="warning",
            logger="test.server",
            data="something went wrong",
        )
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            await _default_logging_callback(params)
            mock_get_logger.assert_called_with("test.server")
            mock_logger.log.assert_called_once()
            # Should use WARNING level
            import logging
            assert mock_logger.log.call_args[0][0] == logging.WARNING

    @pytest.mark.asyncio
    async def test_uses_default_logger_name(self):
        from mcp.types import LoggingMessageNotificationParams

        params = LoggingMessageNotificationParams(
            level="info",
            data="hello",
        )
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            await _default_logging_callback(params)
            mock_get_logger.assert_called_with("mcp.server")
