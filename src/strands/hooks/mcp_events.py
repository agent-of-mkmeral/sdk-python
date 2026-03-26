"""MCP-specific hook events for the Strands Agent SDK.

This module defines events that are emitted during MCP (Model Context Protocol)
operations. These events allow hook providers to observe and react to MCP-specific
lifecycle events such as server connections, tool calls, and server notifications.

Option B (Pass-Through) Design
-------------------------------
These events are designed for use with the pass-through MCP integration pattern.
Users wire callbacks manually on MCPClient instances and can use these events
to build observability, logging, or custom handling on top of the MCP protocol.

No Agent changes are required—these events can be emitted by user code or
future MCPClient extensions without modifying the Agent class.

Usage Example::

    from strands.hooks import HookProvider, HookRegistry
    from strands.hooks.mcp_events import (
        MCPServerConnectedEvent,
        MCPToolCallStartEvent,
        MCPToolCallEndEvent,
    )

    class MCPObserver(HookProvider):
        def register_hooks(self, registry: HookRegistry) -> None:
            registry.add_callback(MCPServerConnectedEvent, self.on_connected)
            registry.add_callback(MCPToolCallStartEvent, self.on_tool_start)
            registry.add_callback(MCPToolCallEndEvent, self.on_tool_end)

        def on_connected(self, event: MCPServerConnectedEvent) -> None:
            print(f"Connected to MCP server: {event.server_name}")

        def on_tool_start(self, event: MCPToolCallStartEvent) -> None:
            print(f"Calling MCP tool: {event.tool_name}")

        def on_tool_end(self, event: MCPToolCallEndEvent) -> None:
            print(f"Tool {event.tool_name} completed: {event.status}")
"""

from dataclasses import dataclass, field
from typing import Any

from .registry import BaseHookEvent


# ==================================================================================
# MCP Server Lifecycle Events
# ==================================================================================


@dataclass
class MCPServerConnectedEvent(BaseHookEvent):
    """Event triggered when an MCP server connection is established.

    Fired after the MCP ClientSession has been initialized and the server
    has responded with its capabilities.

    Attributes:
        server_name: Human-readable name or identifier for the MCP server.
        server_instructions: Optional instructions provided by the server during init.
        capabilities: Raw server capabilities dict from the MCP handshake.
    """

    server_name: str = ""
    server_instructions: str | None = None
    capabilities: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPServerDisconnectedEvent(BaseHookEvent):
    """Event triggered when an MCP server connection is closed.

    Fired when the MCPClient context manager exits or stop() is called.

    Attributes:
        server_name: Human-readable name or identifier for the MCP server.
        error: Optional exception if the disconnection was due to an error.
    """

    server_name: str = ""
    error: Exception | None = None

    @property
    def should_reverse_callbacks(self) -> bool:
        """True to invoke callbacks in reverse order (cleanup pattern)."""
        return True


# ==================================================================================
# MCP Tool Call Events
# ==================================================================================


@dataclass
class MCPToolCallStartEvent(BaseHookEvent):
    """Event triggered before an MCP tool call is executed.

    Fired just before ``session.call_tool()`` is invoked, allowing observers
    to log, modify arguments, or implement pre-call logic.

    Attributes:
        tool_name: Name of the MCP tool being called.
        tool_use_id: Unique identifier for this tool use.
        arguments: Arguments that will be passed to the tool. May be None.
        meta: Protocol-level metadata extracted from ``_meta`` in arguments.
    """

    tool_name: str = ""
    tool_use_id: str = ""
    arguments: dict[str, Any] | None = None
    meta: dict[str, Any] | None = None


@dataclass
class MCPToolCallEndEvent(BaseHookEvent):
    """Event triggered after an MCP tool call completes.

    Fired after ``session.call_tool()`` returns (success or error).

    Attributes:
        tool_name: Name of the MCP tool that was called.
        tool_use_id: Unique identifier for this tool use.
        status: Result status string ("success" or "error").
        is_error: Whether the MCP server reported an error.
        duration_ms: Duration of the tool call in milliseconds, if measured.
        error: Exception if the tool call raised an exception.
    """

    tool_name: str = ""
    tool_use_id: str = ""
    status: str = "success"
    is_error: bool = False
    duration_ms: float | None = None
    error: Exception | None = None

    @property
    def should_reverse_callbacks(self) -> bool:
        """True to invoke callbacks in reverse order (cleanup pattern)."""
        return True


# ==================================================================================
# MCP Server Notification Events
# ==================================================================================


@dataclass
class MCPLoggingEvent(BaseHookEvent):
    """Event triggered when an MCP server sends a log notification.

    Wraps the MCP ``notifications/message`` log messages so hook providers
    can implement custom log routing or aggregation.

    Attributes:
        level: MCP log level string (e.g., "info", "error", "debug").
        logger_name: Optional logger name from the MCP server.
        data: The log message data (can be any JSON-serializable value).
        server_name: Name of the MCP server that sent the log.
    """

    level: str = "info"
    logger_name: str | None = None
    data: Any = None
    server_name: str = ""


@dataclass
class MCPProgressEvent(BaseHookEvent):
    """Event triggered when an MCP tool reports progress.

    Wraps progress notifications from ``session.call_tool()`` so hook providers
    can implement progress bars, logging, or UI updates.

    Attributes:
        tool_name: Name of the tool reporting progress.
        tool_use_id: Unique identifier for this tool use.
        progress: Current progress value.
        total: Total expected value (may be None if unknown).
        message: Optional human-readable progress message.
    """

    tool_name: str = ""
    tool_use_id: str = ""
    progress: float = 0.0
    total: float | None = None
    message: str | None = None


@dataclass
class MCPSamplingRequestEvent(BaseHookEvent):
    """Event triggered when an MCP server requests sampling (LLM completion).

    This event wraps the ``sampling/createMessage`` request from the server,
    allowing hook providers to observe or log sampling requests.

    Attributes:
        server_name: Name of the MCP server requesting sampling.
        model: Requested model identifier, if any.
        max_tokens: Maximum tokens requested.
    """

    server_name: str = ""
    model: str | None = None
    max_tokens: int | None = None
