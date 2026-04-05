from __future__ import annotations

"""
MCP (Model Context Protocol) bridge for agent-forge.

Reads mcp_servers.json at the project root, connects to each server,
discovers its tools, and registers async wrappers in the tool registry.
All registrations are cleaned up on exit so tools don't leak between runs.

Adding a new MCP server:
    1. Edit mcp_servers.json at the project root.
    2. Restart the backend — no code changes needed.

Supported transports:
    - stdio  — launch a local subprocess (command + args)
    - HTTP   — connect to a running SSE server (url)
"""

import json
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# mcp_servers.json lives at the project root (4 levels up from this file)
_CONFIG_PATH = Path(__file__).parents[3] / "mcp_servers.json"


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server.

    Use *either* ``command`` + ``args`` (stdio) *or* ``url`` (HTTP/SSE).

    Attributes:
        name:    Short identifier used as a prefix for all tools from this
                 server, e.g. ``"filesystem"`` → ``mcp_filesystem_read_file``.
        command: Executable to launch for stdio transport (e.g. ``"npx"``).
        args:    Arguments for the stdio executable.
        url:     SSE endpoint URL for HTTP transport.
    """
    name: str
    command: str = ""
    args: list[str] = field(default_factory=list)
    url: str = ""


def load_config(path: Path = _CONFIG_PATH) -> list[MCPServerConfig]:
    """Load MCP server configs from a JSON file.

    Returns an empty list if the file does not exist or is empty.
    """
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [MCPServerConfig(**entry) for entry in data]


class MCPBridge:
    """Async context manager that wires MCP servers into the tool registry.

    On entry  — connects to every configured server, discovers its tools, and
                registers async wrapper callables as ``mcp_{server}_{tool}``.
    On exit   — deregisters all MCP tools and closes every session cleanly.

    Usage::

        async with MCPBridge() as bridge:
            # tool registry now includes all MCP tools
            ...
        # registry is clean again

    Args:
        configs: Server configs to use. Defaults to ``load_config()`` which
                 reads ``mcp_servers.json`` from the project root.
    """

    def __init__(self, configs: list[MCPServerConfig] | None = None) -> None:
        self._configs = configs if configs is not None else load_config()
        self._stack = AsyncExitStack()
        self._registered: list[str] = []

    async def __aenter__(self) -> MCPBridge:
        from mcp import ClientSession, StdioServerParameters
        from agent_forge.tools import register_callable

        await self._stack.__aenter__()

        for cfg in self._configs:
            if cfg.command:
                from mcp.client.stdio import stdio_client
                params = StdioServerParameters(command=cfg.command, args=cfg.args)
                read, write = await self._stack.enter_async_context(stdio_client(params))
            elif cfg.url:
                from mcp.client.sse import sse_client
                read, write = await self._stack.enter_async_context(sse_client(cfg.url))
            else:
                import warnings
                warnings.warn(f"MCP server {cfg.name!r} has no command or url — skipping.")
                continue

            session: ClientSession = await self._stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()

            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                full_name = f"mcp_{cfg.name}_{tool.name}"

                # Capture session + tool name in closure
                def _make_wrapper(s: ClientSession, t_name: str, t_desc: str):
                    async def wrapper(**kwargs: Any) -> str:
                        result = await s.call_tool(t_name, kwargs)
                        return "\n".join(
                            c.text for c in result.content if hasattr(c, "text")
                        )
                    wrapper.__doc__ = t_desc
                    return wrapper

                wrapper = _make_wrapper(session, tool.name, tool.description or "")
                register_callable(full_name, wrapper, tool.description or "")
                self._registered.append(full_name)

        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        from agent_forge.tools import unregister
        unregister(self._registered)
        await self._stack.__aexit__(*exc_info)

    @property
    def tool_names(self) -> list[str]:
        """Names of all MCP tools currently registered by this bridge."""
        return list(self._registered)
