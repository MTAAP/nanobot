"""MCP manager for coordinating multiple MCP servers."""

from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.mcp.client import MCPClient
from nanobot.mcp.protocol import MCPToolInfo

if TYPE_CHECKING:
    from nanobot.config.schema import MCPConfig


class MCPManager:
    """
    Manages multiple MCP server connections.

    Coordinates server lifecycle, tool discovery, and tool execution
    across all configured MCP servers.
    """

    def __init__(self, config: "MCPConfig"):
        """
        Initialize the MCP manager.

        Args:
            config: MCP configuration with server definitions.
        """
        self._config = config
        self._clients: dict[str, MCPClient] = {}
        self._tool_map: dict[str, tuple[str, str]] = {}  # prefixed_name -> (server_name, tool_name)
        self._started = False
        self._server_errors: dict[str, str] = {}

    @property
    def is_enabled(self) -> bool:
        """Check if MCP is enabled in config."""
        return self._config.enabled

    @property
    def server_errors(self) -> dict[str, str]:
        """Get errors from servers that failed to start."""
        return self._server_errors.copy()

    def get_all_tools(self) -> list[tuple[str, MCPToolInfo]]:
        """
        Get all tools from all running servers.

        Returns:
            List of (prefixed_name, tool_info) tuples.
        """
        tools = []
        for server_name, client in self._clients.items():
            if not client.is_running:
                continue

            for tool in client.tools:
                prefixed_name = f"{server_name}_{tool.name}"

                # Check if tool is disabled in config
                tool_config = self._config.tools.get(prefixed_name)
                if tool_config and not tool_config.enabled:
                    continue

                tools.append((prefixed_name, tool))

        return tools

    def get_tool_server(self, prefixed_name: str) -> tuple[str, str] | None:
        """
        Get the server and original tool name for a prefixed tool name.

        Args:
            prefixed_name: Tool name with server prefix (e.g., "github_create_issue").

        Returns:
            Tuple of (server_name, original_tool_name) or None if not found.
        """
        return self._tool_map.get(prefixed_name)

    async def start(self) -> None:
        """Start all configured MCP servers."""
        if self._started or not self._config.enabled:
            return

        self._started = True
        self._server_errors.clear()

        for server_name, server_config in self._config.servers.items():
            if not server_config.enabled:
                logger.debug(f"MCP server {server_name} is disabled, skipping")
                continue

            logger.debug(
                f"MCP server {server_name} config: cmd={server_config.command}, "
                f"args={server_config.args}, env={server_config.env}"
            )
            client = MCPClient(
                name=server_name,
                command=server_config.command,
                args=server_config.args,
                env=server_config.env if server_config.env else None,
                timeout=server_config.timeout,
            )

            success = await client.start()
            if success:
                self._clients[server_name] = client
                # Build tool map
                for tool in client.tools:
                    prefixed_name = f"{server_name}_{tool.name}"
                    self._tool_map[prefixed_name] = (server_name, tool.name)
            else:
                # Store error for diagnosis
                error = client.start_error or f"Unknown error starting {server_name}"
                self._server_errors[server_name] = error
                logger.warning(f"MCP server {server_name} failed to start: {error}")

        running_count = len(self._clients)
        total_tools = len(self._tool_map)
        logger.info(f"MCP manager started: {running_count} servers, {total_tools} tools")

    async def stop(self) -> None:
        """Stop all MCP servers."""
        for client in self._clients.values():
            await client.stop()
        self._clients.clear()
        self._tool_map.clear()
        self._started = False
        logger.debug("MCP manager stopped")

    async def call_tool(self, prefixed_name: str, arguments: dict[str, Any]) -> str:
        """
        Call a tool by its prefixed name.

        Args:
            prefixed_name: Tool name with server prefix (e.g., "github_create_issue").
            arguments: Arguments for the tool.

        Returns:
            Tool result as a string.
        """
        mapping = self._tool_map.get(prefixed_name)
        if not mapping:
            return f"Error: Tool '{prefixed_name}' not found"

        server_name, tool_name = mapping
        client = self._clients.get(server_name)
        if not client:
            return f"Error: MCP server '{server_name}' not running"

        return await client.call_tool(tool_name, arguments)

    def get_server_status(self) -> dict[str, dict[str, Any]]:
        """
        Get status information for all servers.

        Returns:
            Dict mapping server name to status info.
        """
        status = {}

        # Running servers
        for server_name, client in self._clients.items():
            info = client.server_info
            status[server_name] = {
                "running": client.is_running,
                "name": info.name if info else "unknown",
                "version": info.version if info else "unknown",
                "tools": [t.name for t in client.tools],
            }

        # Failed servers
        for server_name, error in self._server_errors.items():
            if server_name not in status:
                status[server_name] = {
                    "running": False,
                    "error": error,
                    "tools": [],
                }

        return status
