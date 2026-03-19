"""First-class MCP integration for Strands Agents.

This package provides high-level utilities for managing multiple MCP server
connections and registering their tools with an Agent.

Key components:

- :class:`MCPRegistry` – manages a collection of :class:`MCPClient` instances,
  registers tools, and routes MCP server notifications to the Strands hook system.
- :func:`load_mcp_servers` – factory that creates :class:`MCPClient` instances
  from a standard ``mcpServers`` JSON configuration dict.
"""

from .registry import MCPRegistry, load_mcp_servers

__all__ = ["MCPRegistry", "load_mcp_servers"]
