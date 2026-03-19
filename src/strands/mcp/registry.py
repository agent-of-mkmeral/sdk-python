"""MCPRegistry – manages MCP clients as first-class Agent citizens.

The registry:
1. Holds named :class:`MCPClient` instances.
2. On :meth:`register_tools`, starts each client and collects their tools
   into the agent's :class:`ToolRegistry`.
3. Installs a unified ``message_handler`` on every client that converts
   MCP server notifications into :mod:`strands.hooks.mcp_events` events
   and dispatches them through the agent's :class:`HookRegistry`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from typing import Any

from ..hooks.mcp_events import (
    MCPLogEvent,
    MCPProgressEvent,
    MCPToolsChangedEvent,
)
from ..tools.mcp.mcp_client import MCPClient, _default_logging_callback

logger = logging.getLogger(__name__)


class MCPRegistry:
    """Manages a set of named MCP server connections.

    Typical usage inside :class:`Agent.__init__`::

        registry = MCPRegistry({"fs": fs_client, "git": git_client})
        registry.register_tools(agent.tool_registry, agent.hooks)
    """

    def __init__(self, clients: dict[str, MCPClient] | None = None) -> None:
        """Initialize the registry.

        Args:
            clients: Mapping of *server_name* → :class:`MCPClient`.
                     The name is used in emitted hook events so subscribers can
                     tell which server a notification came from.
        """
        self._clients: dict[str, MCPClient] = dict(clients) if clients else {}

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def clients(self) -> dict[str, MCPClient]:
        """Return a read-only view of the registered clients."""
        return dict(self._clients)

    @property
    def server_names(self) -> list[str]:
        """Return the names of all registered servers."""
        return list(self._clients.keys())

    def add_client(self, name: str, client: MCPClient) -> None:
        """Add (or replace) a named MCP client.

        Args:
            name: Logical server name.
            client: The :class:`MCPClient` instance.
        """
        self._clients[name] = client

    def register_tools(self, tool_registry: Any, hook_registry: Any) -> None:
        """Start every client and push their tools into *tool_registry*.

        For each client the method also installs a ``logging_callback``
        that wraps the default Python-logging forwarder *and* emits
        :class:`MCPLogEvent` through the hook system.

        Args:
            tool_registry: The agent's :class:`ToolRegistry`.
            hook_registry: The agent's :class:`HookRegistry`.
        """
        from ..tools.registry import ToolRegistry

        if not isinstance(tool_registry, ToolRegistry):
            raise TypeError(f"Expected ToolRegistry, got {type(tool_registry)}")

        for name, client in self._clients.items():
            logger.debug("server_name=<%s> | registering MCP client tools", name)
            # Wrap the existing logging_callback so it also fires hook events
            _install_hook_logging_callback(client, name, hook_registry)
            # Let the ToolRegistry do lazy start + load via process_tools
            tool_registry.process_tools([client])
            logger.debug("server_name=<%s> | MCP client registered", name)


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _install_hook_logging_callback(
    client: MCPClient,
    server_name: str,
    hook_registry: Any,
) -> None:
    """Replace the client's logging_callback with one that also emits hook events.

    The replacement callback:
    1. Calls the *original* logging callback (if any).
    2. Emits an :class:`MCPLogEvent` through the hook registry.
    """
    from mcp.types import LoggingMessageNotificationParams

    original_cb = client._logging_callback

    async def _hook_logging_callback(params: LoggingMessageNotificationParams) -> None:
        # Forward to the original callback first
        if original_cb is not None:
            await original_cb(params)

        level_str = params.level if isinstance(params.level, str) else str(params.level)
        event = MCPLogEvent(
            server_name=server_name,
            level=level_str,
            logger_name=params.logger,
            data=params.data,
        )
        try:
            await hook_registry.invoke_callbacks_async(event)
        except Exception:
            logger.debug("server_name=<%s> | failed to dispatch MCPLogEvent", server_name, exc_info=True)

    client._logging_callback = _hook_logging_callback


# ------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------


def load_mcp_servers(
    config: dict[str, Any] | str,
) -> dict[str, MCPClient]:
    """Create :class:`MCPClient` instances from a standard ``mcpServers`` JSON config.

    The config format matches the *de facto* standard used by Cursor, VS Code
    MCP extensions, and many open-source tools::

        {
          "mcpServers": {
            "filesystem": {
              "command": "npx",
              "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
              "env": {"DEBUG": "1"}
            },
            "remote": {
              "url": "https://mcp.example.com/sse"
            }
          }
        }

    Supported top-level keys inside each server entry:

    - ``command`` / ``args`` / ``env`` – stdio transport.
    - ``url`` / ``headers`` – SSE or Streamable-HTTP transport (auto-detected).
    - ``disabled`` – skip this server (default ``False``).
    - ``prefix`` – tool-name prefix (defaults to the server key).
    - ``disabledTools`` – list of tool names to reject via ``ToolFilters``.

    Args:
        config: Either a *parsed* dict **or** a JSON string.  The dict
                may be the outer wrapper (``{"mcpServers": {…}}``) or the
                inner mapping directly (``{"server_a": {…}, …}``).

    Returns:
        Mapping of *server_name* → :class:`MCPClient`.  Callers can feed
        this directly into :class:`MCPRegistry`.
    """
    if isinstance(config, str):
        config = json.loads(config)

    # Accept both {"mcpServers": {...}} and the inner dict directly.
    # Heuristic: if config has "mcpServers" and its value is a dict whose
    # values are themselves dicts (server configs), treat it as the wrapper.
    # Otherwise, treat the entire config as the inner mapping.
    raw = config.get("mcpServers")
    if isinstance(raw, dict) and all(isinstance(v, dict) for v in raw.values()):
        servers: dict[str, Any] = raw
    else:
        servers = config

    clients: dict[str, MCPClient] = {}
    for name, cfg in servers.items():
        if cfg.get("disabled", False):
            logger.info("server_name=<%s> | skipped (disabled)", name)
            continue

        try:
            client = _build_client_from_config(name, cfg)
            clients[name] = client
            logger.info("server_name=<%s> | created MCPClient", name)
        except Exception:
            logger.exception("server_name=<%s> | failed to create MCPClient", name)

    return clients


def _build_client_from_config(name: str, cfg: dict[str, Any]) -> MCPClient:
    """Build a single :class:`MCPClient` from a server config entry."""
    from ..tools.mcp.mcp_client import ToolFilters

    prefix = cfg.get("prefix", name)

    disabled_tools = cfg.get("disabledTools", [])
    tool_filters: ToolFilters | None = ToolFilters(rejected=disabled_tools) if disabled_tools else None

    if "command" in cfg:
        command = cfg["command"]
        args = cfg.get("args", [])
        env = cfg.get("env")

        from mcp import StdioServerParameters, stdio_client

        def _stdio_transport(
            _cmd: str = command,
            _args: list[str] = args,
            _env: dict[str, str] | None = env,
        ):  # type: ignore[no-untyped-def]
            return stdio_client(StdioServerParameters(command=_cmd, args=_args, env=_env))

        return MCPClient(
            transport_callable=_stdio_transport,
            prefix=prefix,
            tool_filters=tool_filters,
        )

    elif "url" in cfg:
        url: str = cfg["url"]
        headers = cfg.get("headers")

        def _http_transport(_url: str = url, _headers: dict[str, str] | None = headers):  # type: ignore[no-untyped-def]
            if "/sse" in _url:
                from mcp.client.sse import sse_client

                return sse_client(_url)
            else:
                from mcp.client.streamable_http import streamablehttp_client

                return streamablehttp_client(url=_url, headers=_headers)

        return MCPClient(
            transport_callable=_http_transport,
            prefix=prefix,
            tool_filters=tool_filters,
        )

    else:
        raise ValueError(f"Server '{name}' must specify either 'command' or 'url'")
