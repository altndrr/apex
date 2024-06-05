from collections.abc import Callable

import torch
from omegaconf import DictConfig

__all__ = [
    "BUILTIN_TOOLS",
    "get_tool",
    "get_tool_builder",
    "get_tools_docstrings",
    "list_tools",
    "register_tool",
]


AVAIL_CATEGORIES = ["select", "transform"]
BUILTIN_TOOLS = {}
BUILTIN_TOOLS_CATEGORIES = {}


def get_tool(name: str, **tool_cfg: DictConfig | dict) -> torch.nn.Module:
    """Get a tool.

    Args:
    ----
        name (str): The name of the tool.
        tool_cfg (DictConfig | dict): Additional arguments to pass to the tool.

    """
    fn = get_tool_builder(name)
    return fn(**tool_cfg)


def get_tool_builder(name: str) -> Callable:
    """Get the builder of a tool.

    Args:
    ----
        name (str): The name of the tool.

    """
    name = name.lower()
    if name not in BUILTIN_TOOLS:
        raise ValueError(f"Unknown tool {name}. Available tools: {list_tools()}")

    return BUILTIN_TOOLS[name]


def get_tools_docstrings() -> str:
    """Get the documentation of all available tools."""
    docs = []
    for category in AVAIL_CATEGORIES:
        if not any(cat == category for cat in BUILTIN_TOOLS_CATEGORIES.values()):
            continue

        docs.append(f"\n{category.upper()} TOOLS")
        docs.append("")
        for name, tool in BUILTIN_TOOLS.items():
            if BUILTIN_TOOLS_CATEGORIES[name] != category:
                continue
            docs.append(f"{tool.__module__}.{tool.__name__}: {tool.__doc__.replace('    ', '')}")

    return "\n".join(docs).strip()


def list_tools(category: str | None = None) -> list:
    """List all available tools.

    Args:
    ----
        category (str, optional): The category of the tools to list. Defaults to None.

    """
    if category:
        if category not in AVAIL_CATEGORIES:
            raise ValueError(
                f"Unknown tool category {category}. Available categories: {AVAIL_CATEGORIES}"
            )
        return [name for name, cat in BUILTIN_TOOLS_CATEGORIES.items() if cat == category]

    return list(BUILTIN_TOOLS.keys())


def register_tool(name: str | None = None, category: str | None = None) -> Callable:
    """Register a tool.

    Args:
    ----
        name (str, optional): The name of the tool.
        category (str, optional): The category of the tool.

    """
    if category is None:
        raise ValueError("Undefined tool category. Available categories: {AVAIL_CATEGORIES}")

    def decorator(tool: torch.nn.Module) -> torch.nn.Module:
        BUILTIN_TOOLS[name or tool.__name__.lower()] = tool
        if category:
            if category not in AVAIL_CATEGORIES:
                raise ValueError(
                    f"Unknown tool category {category}. Available categories: {AVAIL_CATEGORIES}"
                )

            BUILTIN_TOOLS_CATEGORIES[name or tool.__name__.lower()] = category
        return tool

    return decorator
