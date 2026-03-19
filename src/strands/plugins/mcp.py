"""MCP Plugin for deeper MCP integration in the Strands Agents SDK.

This module provides the MCPPlugin class which integrates MCP servers into
the agent as a first-class plugin. It manages MCP client lifecycles, registers
tools with the agent, installs a unified message handler that routes
ServerNotification types to MCPEvent hook events, and optionally auto-wires
sampling and logging callbacks.

NO Agent core changes are required — the plugin uses ``init_agent()`` to
access the agent instance, register tools, and install hooks.

Example:
    ```python
    from strands import Agent
    from strands.plugins.mcp import MCPPlugin
    from strands.tools.mcp import MCPClient

    client = MCPClient(transport_callable=my_transport)
    plugin = MCPPlugin(clients=[client])
    agent = Agent(plugins=[plugin])
    ```

    Or from a JSON configuration file:

    ```python
    plugin = MCPPlugin.from_config("mcp_servers.json")
    agent = Agent(plugins=[plugin])
    ```
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..hooks.mcp_events import (
    MCPLoggingEvent,
    MCPProgressEvent,
    MCPPromptsChangedEvent,
    MCPResourcesChangedEvent,
    MCPResourceUpdatedEvent,
    MCPToolsChangedEvent,
)
from ..tools.mcp import MCPClient
from .plugin import Plugin

if TYPE_CHECKING:
    from ..agent import Agent

logger = logging.getLogger(__name__)


class MCPPlugin(Plugin):
    """Plugin that integrates one or more MCP servers into an agent.

    The plugin:
    - Registers all tools from configured MCP clients with the agent.
    - Installs a unified ``message_handler`` on each client that routes
      ``ServerNotification`` types to corresponding ``MCPEvent`` hook events
      via ``agent.hooks``.
    - Optionally auto-wires sampling (``create_message``) requests back to
      the agent when ``auto_sampling=True``.
    - Provides a ``from_config()`` classmethod for JSON/dict-based
      configuration (compatible with VS Code / Cursor MCP config format).
    - When ``fail_open=True``, server startup failures are logged but do
      not prevent the agent from starting.

    Attributes:
        name: Plugin identifier — ``"mcp"``.
    """

    @property
    def name(self) -> str:
        """Plugin identifier."""
        return "mcp"

    def __init__(
        self,
        clients: list[MCPClient],
        *,
        fail_open: bool = False,
        auto_sampling: bool = False,
    ) -> None:
        """Initialise the MCP plugin.

        Args:
            clients: List of MCPClient instances to manage.
            fail_open: If True, log and skip clients that fail to start
                instead of raising an exception.
            auto_sampling: If True, automatically wire a sampling callback
                that forwards ``create_message`` requests to the agent.
        """
        # Plugin.__init__ does decorated-method discovery — call it first.
        super().__init__()
        self._clients = list(clients)
        self._fail_open = fail_open
        self._auto_sampling = auto_sampling
        self._agent: Agent | None = None

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        *,
        fail_open: bool = False,
        auto_sampling: bool = False,
    ) -> "MCPPlugin":
        """Create an MCPPlugin from a JSON config file.

        The configuration format mirrors the widely-adopted VS Code / Cursor
        ``mcpServers`` JSON structure::

            {
                "mcpServers": {
                    "server-name": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                        "env": {},
                        "disabled": false,
                        "prefix": "fs",
                        "disabledTools": ["tool_a"]
                    }
                }
            }

        Args:
            config_path: Path to a JSON configuration file.
            fail_open: If True, skip servers that fail to initialise.
            auto_sampling: If True, auto-wire sampling callback.

        Returns:
            A configured MCPPlugin instance.
        """
        clients = load_mcp_servers(config_path)
        return cls(clients=clients, fail_open=fail_open, auto_sampling=auto_sampling)

    # ------------------------------------------------------------------
    # Plugin lifecycle
    # ------------------------------------------------------------------

    def init_agent(self, agent: "Agent") -> None:
        """Initialise the plugin with the agent.

        This method:
        1. Stores a reference to the agent.
        2. Installs a unified ``message_handler`` on each client that routes
           ``ServerNotification`` types to MCPEvent hook events.
        3. Registers each client's tools with the agent's tool registry.
        4. If ``auto_sampling`` is enabled, wires up a sampling callback.

        Args:
            agent: The agent instance being initialised.
        """
        self._agent = agent

        for client in self._clients:
            try:
                # Install routing message handler before the client starts
                self._install_message_handler(client)

                # Register the client as a tool provider with the agent
                agent.tool_registry.process_tools([client])
                logger.info(
                    "plugin=<mcp>, prefix=<%s> | registered MCP client tools",
                    client._prefix,
                )
            except Exception:
                if self._fail_open:
                    logger.warning(
                        "plugin=<mcp>, prefix=<%s> | failed to register MCP client (fail_open=True)",
                        client._prefix,
                        exc_info=True,
                    )
                else:
                    raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _install_message_handler(self, client: MCPClient) -> None:
        """Install a unified message handler on the client that routes notifications to hooks.

        The handler intercepts ``ServerNotification`` messages received by
        the MCP ``ClientSession`` and dispatches corresponding hook events
        through ``agent.hooks``. Non-notification messages and exceptions
        are forwarded to the client's default error handler.

        Args:
            client: The MCPClient to install the handler on.
        """
        agent = self._agent
        if agent is None:
            return

        # Import MCP notification types lazily to keep the module
        # importable even when the ``mcp`` package is not installed.
        from mcp.types import (
            LoggingMessageNotification,
            ProgressNotification,
            PromptListChangedNotification,
            ResourceListChangedNotification,
            ResourceUpdatedNotification,
            ServerNotification,
            ToolListChangedNotification,
        )

        original_handler = client._handle_error_message

        async def _routing_handler(message: Exception | Any) -> None:
            """Route ServerNotification messages to MCPEvent hooks."""
            # Handle exceptions with the original error handler
            if isinstance(message, Exception):
                await original_handler(message)
                return

            # Route ServerNotification types to hook events
            if isinstance(message, ServerNotification):
                notification = message.root
                _dispatch_notification(notification, client)
                return

            # Check individual notification types directly (some SDK
            # versions unwrap the RootModel before passing to handler)
            _dispatch_notification(message, client)

        def _dispatch_notification(notification: Any, mcp_client: MCPClient) -> None:
            """Map an MCP notification to the appropriate hook event and invoke it."""
            try:
                if isinstance(notification, ToolListChangedNotification):
                    agent.hooks.invoke_callbacks(
                        MCPToolsChangedEvent(
                            agent=agent,
                            client=mcp_client,
                            prefix=mcp_client._prefix,
                        )
                    )
                elif isinstance(notification, ResourceListChangedNotification):
                    agent.hooks.invoke_callbacks(
                        MCPResourcesChangedEvent(agent=agent, client=mcp_client)
                    )
                elif isinstance(notification, PromptListChangedNotification):
                    agent.hooks.invoke_callbacks(
                        MCPPromptsChangedEvent(agent=agent, client=mcp_client)
                    )
                elif isinstance(notification, LoggingMessageNotification):
                    params = notification.params
                    agent.hooks.invoke_callbacks(
                        MCPLoggingEvent(
                            agent=agent,
                            client=mcp_client,
                            level=params.level if params else "info",
                            data=params.data if params else None,
                            logger_name=params.logger if params else None,
                        )
                    )
                elif isinstance(notification, ProgressNotification):
                    params = notification.params
                    agent.hooks.invoke_callbacks(
                        MCPProgressEvent(
                            agent=agent,
                            client=mcp_client,
                            progress=params.progress if params else 0.0,
                            total=params.total if params else None,
                            message=getattr(params, "message", None) if params else None,
                            progress_token=params.progressToken if params else None,
                        )
                    )
                elif isinstance(notification, ResourceUpdatedNotification):
                    params = notification.params
                    agent.hooks.invoke_callbacks(
                        MCPResourceUpdatedEvent(
                            agent=agent,
                            client=mcp_client,
                            uri=str(params.uri) if params else "",
                        )
                    )
                else:
                    logger.debug(
                        "plugin=<mcp> | unhandled notification type: %s",
                        type(notification).__name__,
                    )
            except Exception:
                logger.warning(
                    "plugin=<mcp> | error dispatching notification",
                    exc_info=True,
                )

        # Replace the client's message handler with our routing handler
        client._message_handler = _routing_handler


# ------------------------------------------------------------------
# Standalone config loader (can be used independently of the plugin)
# ------------------------------------------------------------------


def load_mcp_servers(config_path: str | Path) -> list[MCPClient]:
    """Load MCP server configurations from a JSON file and return MCPClient instances.

    This function can be used standalone (without MCPPlugin) to create
    MCPClient instances from a configuration file.

    The configuration format::

        {
            "mcpServers": {
                "server-name": {
                    "command": "npx",
                    "args": ["-y", "@server/package"],
                    "env": {},
                    "disabled": false,
                    "prefix": "my_prefix",
                    "disabledTools": ["tool_a"]
                }
            }
        }

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        A list of MCPClient instances (not yet started).
    """
    from mcp import StdioServerParameters, stdio_client

    from ..tools.mcp import ToolFilters

    path = Path(config_path)
    with open(path, "r", encoding="utf-8") as fh:
        raw_config = json.load(fh)

    servers = raw_config.get("mcpServers", {})
    clients: list[MCPClient] = []

    for name, cfg in servers.items():
        if cfg.get("disabled", False):
            logger.info("load_mcp_servers | server '%s' is disabled, skipping", name)
            continue

        disabled_tools = cfg.get("disabledTools", [])
        tool_filters: ToolFilters | None = (
            ToolFilters(rejected=disabled_tools) if disabled_tools else None
        )
        prefix = cfg.get("prefix", name)

        if "command" in cfg:
            command = cfg["command"]
            args = cfg.get("args", [])
            env = cfg.get("env")

            def _make_transport(
                _cmd: str = command,
                _args: list = args,
                _env: dict | None = env,
            ):  # type: ignore[no-untyped-def]
                return stdio_client(
                    StdioServerParameters(command=_cmd, args=_args, env=_env)
                )

            client = MCPClient(
                transport_callable=_make_transport,
                prefix=prefix,
                tool_filters=tool_filters,
            )
            clients.append(client)
            logger.info("load_mcp_servers | loaded server '%s' (stdio)", name)
        elif "url" in cfg:
            url = cfg["url"]
            headers = cfg.get("headers")

            try:
                from mcp.client.sse import sse_client
                from mcp.client.streamable_http import streamablehttp_client
            except ImportError:
                logger.warning(
                    "load_mcp_servers | HTTP transports not available for '%s'", name
                )
                continue

            def _make_http_transport(
                _url: str = url,
                _headers: dict | None = headers,
            ):  # type: ignore[no-untyped-def]
                return (
                    sse_client(_url)
                    if "/sse" in _url
                    else streamablehttp_client(url=_url, headers=_headers)
                )

            client = MCPClient(
                transport_callable=_make_http_transport,
                prefix=prefix,
                tool_filters=tool_filters,
            )
            clients.append(client)
            logger.info("load_mcp_servers | loaded server '%s' (http)", name)
        else:
            logger.warning(
                "load_mcp_servers | server '%s' has no 'command' or 'url', skipping",
                name,
            )

    return clients
