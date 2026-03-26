"""Model Context Protocol (MCP) integration.

This package provides integration with the Model Context Protocol (MCP), allowing agents to use tools provided by MCP
servers.

- Docs: https://www.anthropic.com/news/model-context-protocol

Option B (Pass-Through) Additions:
- ``load_mcp_servers()``: Load MCP servers from JSON config files/dicts.
- Callback wiring: All MCP ClientSession callbacks are exposed on MCPClient.
- ``_default_logging_callback``: Routes MCP server logs to Python logging.
"""

from .mcp_agent_tool import MCPAgentTool
from .mcp_client import MCPClient, ToolFilters, _default_logging_callback
from .mcp_config import load_mcp_servers
from .mcp_tasks import TasksConfig
from .mcp_types import MCPToolResult, MCPTransport

__all__ = [
    "MCPAgentTool",
    "MCPClient",
    "MCPToolResult",
    "MCPTransport",
    "TasksConfig",
    "ToolFilters",
    "load_mcp_servers",
    "_default_logging_callback",
]
