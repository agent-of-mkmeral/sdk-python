"""MCP server configuration loading utilities.

This module provides utility functions for loading MCP server configurations
from JSON files or dictionaries, following the standard MCP configuration format
used by tools like Claude Desktop, Cursor, and other MCP hosts.

Option B (Pass-Through) Design
-------------------------------
This utility creates MCPClient instances with pass-through callback wiring.
Users can provide callbacks at the config level (applied to all servers) or
override per-server. No Agent-level changes are needed.

Configuration Format::

    {
        "mcpServers": {
            "my-server": {
                "command": "npx",
                "args": ["-y", "@my/mcp-server"],
                "env": {"API_KEY": "..."},
                "disabled": false,
                "disabledTools": ["tool_to_skip"]
            },
            "remote-server": {
                "url": "https://example.com/mcp",
                "headers": {"Authorization": "Bearer ..."}
            }
        }
    }

Usage Example::

    from strands.tools.mcp import load_mcp_servers

    # From a JSON file
    clients = load_mcp_servers("/path/to/mcp.json")

    # From a dict
    config = {"mcpServers": {"my-server": {"command": "npx", "args": [...]}}}
    clients = load_mcp_servers(config)

    # With callbacks applied to all servers
    clients = load_mcp_servers(config, logging_callback=my_logger)

    # Use with Agent
    agent = Agent(tools=[*clients])
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mcp.client.session import ElicitationFnT, ListRootsFnT, LoggingFnT, SamplingFnT
from mcp.shared.session import ProgressFnT

from .mcp_client import MCPClient, ToolFilters

logger = logging.getLogger(__name__)


def load_mcp_servers(
    config: str | Path | dict[str, Any],
    *,
    sampling_callback: SamplingFnT | None = None,
    list_roots_callback: ListRootsFnT | None = None,
    logging_callback: LoggingFnT | None = None,
    progress_callback: ProgressFnT | None = None,
    elicitation_callback: ElicitationFnT | None = None,
    startup_timeout: int = 30,
) -> list[MCPClient]:
    """Load MCP server configurations and create MCPClient instances.

    Accepts a JSON file path, a JSON string, or a parsed dict. The configuration
    follows the standard MCP host format with ``mcpServers`` as the top-level key.

    Each server entry can specify:

    - ``command`` + ``args`` + ``env``: For stdio-based servers.
    - ``url`` + ``headers``: For HTTP/SSE-based servers.
    - ``disabled``: Boolean to skip the server entirely.
    - ``disabledTools``: List of tool names to reject.
    - ``prefix``: Optional prefix for tool names (defaults to server key).

    Callbacks provided to this function are applied to **all** created MCPClient
    instances, enabling shared logging, sampling, or progress handling.

    Args:
        config: Path to JSON file, JSON string, or parsed dict.
        sampling_callback: Sampling callback applied to all servers.
        list_roots_callback: List roots callback applied to all servers.
        logging_callback: Logging callback applied to all servers.
            Pass ``None`` to use the MCPClient default (Python logging).
        progress_callback: Progress callback applied to all servers.
        elicitation_callback: Elicitation callback applied to all servers.
        startup_timeout: Startup timeout for each MCPClient.

    Returns:
        List of MCPClient instances (not yet started—use as context managers
        or pass to Agent's tools parameter).

    Raises:
        ValueError: If config format is invalid or no ``mcpServers`` key found.
        FileNotFoundError: If config is a file path that doesn't exist.

    Example::

        # Load from file
        clients = load_mcp_servers("~/.config/mcp/servers.json")
        agent = Agent(tools=clients)

        # Load from dict with shared progress callback
        def on_progress(progress, total, message):
            print(f"{progress}/{total}: {message}")

        clients = load_mcp_servers(
            {"mcpServers": {"my-server": {"command": "my-cmd"}}},
            progress_callback=on_progress,
        )
    """
    # Parse config
    parsed_config = _parse_config(config)

    servers = parsed_config.get("mcpServers", {})
    if not servers:
        logger.warning("no mcpServers found in configuration")
        return []

    clients: list[MCPClient] = []

    for name, server_cfg in servers.items():
        try:
            # Skip disabled servers
            if server_cfg.get("disabled", False):
                logger.info("server=%s | skipping disabled server", name)
                continue

            # Build tool filters from disabledTools
            disabled_tools = server_cfg.get("disabledTools", [])
            tool_filters: ToolFilters | None = ToolFilters(rejected=disabled_tools) if disabled_tools else None

            # Determine prefix
            prefix = server_cfg.get("prefix", name)

            # Build transport callable
            transport_callable = _build_transport(name, server_cfg)
            if transport_callable is None:
                logger.warning("server=%s | no valid transport configuration found, skipping", name)
                continue

            # Determine logging callback: use provided or sentinel for "use default"
            effective_logging = logging_callback if logging_callback is not None else None

            # Build kwargs, only passing logging_callback if explicitly provided
            kwargs: dict[str, Any] = {
                "startup_timeout": startup_timeout,
                "tool_filters": tool_filters,
                "prefix": prefix,
                "sampling_callback": sampling_callback,
                "list_roots_callback": list_roots_callback,
                "progress_callback": progress_callback,
                "elicitation_callback": elicitation_callback,
            }
            if logging_callback is not None:
                kwargs["logging_callback"] = logging_callback

            client = MCPClient(
                transport_callable=transport_callable,
                **kwargs,
            )
            clients.append(client)
            logger.info("server=%s | created MCPClient", name)

        except Exception as e:
            logger.error("server=%s, error=%s | failed to create MCPClient", name, e)

    logger.info("loaded %d MCP server(s)", len(clients))
    return clients


def _parse_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Parse configuration from various input formats.

    Args:
        config: Path to JSON file, JSON string, or parsed dict.

    Returns:
        Parsed configuration dict.

    Raises:
        ValueError: If config format is invalid.
        FileNotFoundError: If file path doesn't exist.
    """
    if isinstance(config, dict):
        return config

    if isinstance(config, Path):
        config = str(config)

    if isinstance(config, str):
        # Try as file path first
        config_path = Path(config).expanduser()
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # Try as JSON string
        try:
            return json.loads(config)
        except json.JSONDecodeError as e:
            raise ValueError(f"config is not a valid file path or JSON string: {e}") from e

    raise ValueError(f"unsupported config type: {type(config)}")


def _build_transport(name: str, server_cfg: dict[str, Any]) -> Callable | None:
    """Build a transport callable from server configuration.

    Args:
        name: Server name (for logging).
        server_cfg: Server configuration dict.

    Returns:
        Transport callable, or None if no valid transport config found.
    """
    if "command" in server_cfg:
        command = server_cfg["command"]
        args = server_cfg.get("args", [])
        env = server_cfg.get("env")

        def _stdio_transport(
            _cmd: str = command,
            _args: list = args,
            _env: dict | None = env,
        ):  # type: ignore[no-untyped-def]
            from mcp import StdioServerParameters, stdio_client

            return stdio_client(StdioServerParameters(command=_cmd, args=_args, env=_env))

        return _stdio_transport

    elif "url" in server_cfg:
        url = server_cfg["url"]
        headers = server_cfg.get("headers")

        def _http_transport(
            _url: str = url,
            _headers: dict | None = headers,
        ):  # type: ignore[no-untyped-def]
            from mcp.client.sse import sse_client
            from mcp.client.streamable_http import streamablehttp_client

            if "/sse" in _url:
                return sse_client(_url)
            else:
                return streamablehttp_client(url=_url, headers=_headers)

        return _http_transport

    return None
