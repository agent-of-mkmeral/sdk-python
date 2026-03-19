"""Tests for MCP hook events."""

import pytest

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
from strands.hooks.registry import BaseHookEvent


class TestMCPEventHierarchy:
    """Verify inheritance and field defaults."""

    def test_mcp_event_is_base_hook_event(self):
        assert issubclass(MCPEvent, BaseHookEvent)

    def test_mcp_log_event_fields(self):
        evt = MCPLogEvent(server_name="srv", level="warning", logger_name="mylogger", data="boom")
        assert evt.server_name == "srv"
        assert evt.level == "warning"
        assert evt.logger_name == "mylogger"
        assert evt.data == "boom"

    def test_mcp_log_event_defaults(self):
        evt = MCPLogEvent(server_name="s")
        assert evt.level == "info"
        assert evt.logger_name is None
        assert evt.data is None

    def test_mcp_progress_event_fields(self):
        evt = MCPProgressEvent(server_name="s", progress=50, total=100, message="halfway")
        assert evt.progress == 50
        assert evt.total == 100
        assert evt.message == "halfway"

    def test_mcp_progress_event_defaults(self):
        evt = MCPProgressEvent(server_name="s")
        assert evt.progress == 0.0
        assert evt.total is None
        assert evt.message is None

    def test_mcp_tools_changed_event(self):
        evt = MCPToolsChangedEvent(server_name="s")
        assert evt.server_name == "s"

    def test_mcp_prompts_changed_event(self):
        evt = MCPPromptsChangedEvent(server_name="s")
        assert evt.server_name == "s"

    def test_mcp_resources_changed_event(self):
        evt = MCPResourcesChangedEvent(server_name="s")
        assert evt.server_name == "s"

    def test_mcp_resource_updated_event(self):
        evt = MCPResourceUpdatedEvent(server_name="s", uri="file:///tmp/x.txt")
        assert evt.uri == "file:///tmp/x.txt"

    def test_mcp_cancelled_event(self):
        evt = MCPCancelledEvent(server_name="s", request_id="req-1", reason="timeout")
        assert evt.request_id == "req-1"
        assert evt.reason == "timeout"

    def test_mcp_cancelled_event_defaults(self):
        evt = MCPCancelledEvent(server_name="s")
        assert evt.request_id is None
        assert evt.reason is None

    def test_all_events_are_mcp_events(self):
        """All concrete MCP event classes should be subclasses of MCPEvent."""
        for cls in [
            MCPLogEvent,
            MCPProgressEvent,
            MCPToolsChangedEvent,
            MCPPromptsChangedEvent,
            MCPResourcesChangedEvent,
            MCPResourceUpdatedEvent,
            MCPCancelledEvent,
        ]:
            assert issubclass(cls, MCPEvent), f"{cls.__name__} should inherit MCPEvent"
