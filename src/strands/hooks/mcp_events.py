"""MCP-specific hook events for the Strands Agent SDK.

This module defines events emitted by MCP (Model Context Protocol) server connections,
allowing hooks to observe and react to MCP-level notifications such as logging messages,
progress updates, tool list changes, and resource updates.

These events integrate MCP server notifications into the standard Strands hook system,
enabling users to build observability, caching invalidation, and custom routing logic
on top of MCP connections.
"""

from dataclasses import dataclass, field
from typing import Any

from .registry import BaseHookEvent


@dataclass
class MCPEvent(BaseHookEvent):
    """Base class for all MCP-related hook events.

    Attributes:
        server_name: The logical name of the MCP server that emitted this event.
    """

    server_name: str


@dataclass
class MCPLogEvent(MCPEvent):
    """Event emitted when an MCP server sends a logging notification.

    Attributes:
        level: The log level (e.g. "debug", "info", "warning", "error", "critical",
               "alert", "emergency").
        logger_name: Optional name of the logger on the server side.
        data: The log payload (string, dict, or any JSON-serializable value).
    """

    level: str = "info"
    logger_name: str | None = None
    data: Any = None


@dataclass
class MCPProgressEvent(MCPEvent):
    """Event emitted when an MCP tool call reports progress.

    Attributes:
        progress: Current progress value.
        total: Optional total value for computing percentage.
        message: Optional human-readable progress message.
    """

    progress: float = 0.0
    total: float | None = None
    message: str | None = None


@dataclass
class MCPToolsChangedEvent(MCPEvent):
    """Event emitted when the MCP server's tool list has changed.

    Consumers should re-fetch tools from the server when they receive this event.
    """

    pass


@dataclass
class MCPPromptsChangedEvent(MCPEvent):
    """Event emitted when the MCP server's prompt list has changed."""

    pass


@dataclass
class MCPResourcesChangedEvent(MCPEvent):
    """Event emitted when the MCP server's resource list has changed."""

    pass


@dataclass
class MCPResourceUpdatedEvent(MCPEvent):
    """Event emitted when a specific resource on the MCP server has been updated.

    Attributes:
        uri: The URI of the resource that was updated.
    """

    uri: str = ""


@dataclass
class MCPCancelledEvent(MCPEvent):
    """Event emitted when an MCP request or operation was cancelled.

    Attributes:
        request_id: Optional identifier of the cancelled request.
        reason: Optional human-readable reason for cancellation.
    """

    request_id: str | None = None
    reason: str | None = None
