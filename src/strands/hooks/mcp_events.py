"""MCP-specific hook events for the Strands Agent SDK.

This module defines hook events that are emitted by the MCPPlugin when MCP
server notifications are received. These events allow hook providers to
react to MCP server lifecycle changes without modifying the Agent core.

The event hierarchy mirrors the MCP ServerNotification types:

- MCPToolsChangedEvent: Tools list was updated
- MCPResourcesChangedEvent: Resources list was updated
- MCPPromptsChangedEvent: Prompts list was updated
- MCPLoggingEvent: Server sent a log message
- MCPProgressEvent: Server sent a progress notification
- MCPResourceUpdatedEvent: A specific resource was updated

Example:
    ```python
    from strands.hooks import HookProvider, HookRegistry
    from strands.hooks.mcp_events import MCPLoggingEvent

    class MCPLogger(HookProvider):
        def register_hooks(self, registry: HookRegistry) -> None:
            registry.add_callback(MCPLoggingEvent, self.on_log)

        def on_log(self, event: MCPLoggingEvent) -> None:
            print(f"[MCP:{event.logger_name}] {event.level}: {event.data}")
    ```
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .registry import HookEvent

if TYPE_CHECKING:
    from ..tools.mcp import MCPClient


@dataclass
class MCPToolsChangedEvent(HookEvent):
    """Event fired when an MCP server signals that its tool list changed.

    Hook providers can use this to dynamically refresh available tools.

    Attributes:
        client: The MCPClient that received the notification.
        prefix: The tool prefix associated with this client, if any.
    """

    client: "MCPClient" = field(default=None)  # type: ignore[assignment]
    prefix: str | None = None


@dataclass
class MCPResourcesChangedEvent(HookEvent):
    """Event fired when an MCP server signals that its resource list changed.

    Attributes:
        client: The MCPClient that received the notification.
    """

    client: "MCPClient" = field(default=None)  # type: ignore[assignment]


@dataclass
class MCPPromptsChangedEvent(HookEvent):
    """Event fired when an MCP server signals that its prompt list changed.

    Attributes:
        client: The MCPClient that received the notification.
    """

    client: "MCPClient" = field(default=None)  # type: ignore[assignment]


@dataclass
class MCPLoggingEvent(HookEvent):
    """Event fired when an MCP server sends a logging notification.

    Attributes:
        client: The MCPClient that received the notification.
        level: The MCP log level (e.g. "info", "warning", "error").
        data: The log message payload (can be any JSON-serializable value).
        logger_name: Optional logger name from the server.
    """

    client: "MCPClient" = field(default=None)  # type: ignore[assignment]
    level: str = "info"
    data: Any = None
    logger_name: str | None = None


@dataclass
class MCPProgressEvent(HookEvent):
    """Event fired when an MCP server sends a progress notification.

    Attributes:
        client: The MCPClient that received the notification.
        progress: Current progress value.
        total: Total progress value (may be None for indeterminate progress).
        message: Optional human-readable progress message.
        progress_token: The token identifying which operation this progress is for.
    """

    client: "MCPClient" = field(default=None)  # type: ignore[assignment]
    progress: float = 0.0
    total: float | None = None
    message: str | None = None
    progress_token: str | int | None = None


@dataclass
class MCPResourceUpdatedEvent(HookEvent):
    """Event fired when an MCP server signals that a specific resource was updated.

    Attributes:
        client: The MCPClient that received the notification.
        uri: The URI of the updated resource.
    """

    client: "MCPClient" = field(default=None)  # type: ignore[assignment]
    uri: str = ""
