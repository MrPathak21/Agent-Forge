from __future__ import annotations

"""
Tool registry for agent-forge.

Tools self-register via the @register decorator when their module is imported.
The registry is lazy-loaded — all tool modules are imported on first access.
"""

import importlib
import inspect
from typing import Callable

_REGISTRY: dict[str, Callable] = {}
_LOADED = False

# All tool modules — add new ones here to make them discoverable
_MODULES = [
    "agent_forge.tools.utility",
    "agent_forge.tools.web",
    "agent_forge.tools.finance",
]


def register(name: str, description: str = ""):
    """Decorator that registers a function as a named tool."""
    def decorator(fn: Callable) -> Callable:
        setattr(fn, "_tool_name", name)
        setattr(fn, "_tool_description", description or fn.__doc__ or "")
        _REGISTRY[name] = fn
        return fn
    return decorator


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    for mod in _MODULES:
        importlib.import_module(mod)
    _LOADED = True


def list_tools() -> list[str]:
    """Return names of all available tools."""
    _ensure_loaded()
    return sorted(_REGISTRY.keys())


def get_tool(name: str) -> Callable:
    """Look up a single tool by name. Raises KeyError if not found."""
    _ensure_loaded()
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown tool: {name!r}. Available: {list_tools()}"
        )
    return _REGISTRY[name]


def get_tools(names: list[str]) -> list[Callable]:
    """
    Resolve a list of tool names to their callable functions.
    Unknown names are silently skipped with a warning.
    """
    _ensure_loaded()
    resolved = []
    for name in names:
        if name in _REGISTRY:
            resolved.append(_REGISTRY[name])
        else:
            import warnings
            warnings.warn(f"Tool {name!r} not found in registry — skipping.")
    return resolved


_TYPE_MAP: dict = {int: "integer", float: "number", bool: "boolean", str: "string"}


def to_openai_schema(fn: Callable) -> dict:
    """Convert a registered tool function to OpenAI's function calling schema."""
    sig = inspect.signature(fn)
    doc = inspect.getdoc(fn) or ""
    name = getattr(fn, "_tool_name", fn.__name__)

    properties: dict = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        json_type = _TYPE_MAP.get(param.annotation, "string")
        properties[param_name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": doc,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def get_openai_schemas(names: list[str]) -> list[dict]:
    """Return OpenAI function schemas for the given tool names."""
    _ensure_loaded()
    return [to_openai_schema(_REGISTRY[n]) for n in names if n in _REGISTRY]


def register_callable(name: str, fn: Callable, description: str = "") -> None:
    """Register an already-created callable directly.

    Used by MCPBridge to inject MCP tool wrappers at runtime without
    going through the ``@register`` decorator.
    """
    setattr(fn, "_tool_name", name)
    setattr(fn, "_tool_description", description)
    _REGISTRY[name] = fn


def unregister(names: list[str]) -> None:
    """Remove tool names from the registry.

    Used by MCPBridge on exit to clean up MCP tools so they don't
    persist between pipeline runs.
    """
    for name in names:
        _REGISTRY.pop(name, None)
