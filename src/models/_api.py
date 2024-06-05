from collections.abc import Callable

import torch
from omegaconf import DictConfig

__all__ = ["BUILTIN_MODELS", "get_model", "get_model_builder", "list_models", "register_model"]

BUILTIN_MODELS = {}


def get_model(name: str, **model_cfg: DictConfig | dict) -> torch.nn.Module:
    """Get the model name and configuration and returns an instantiated model.

    Args:
    ----
        name (str): The name under which the model is registered.
        model_cfg (DictConfig | dict): parameters passed to the model builder method.

    """
    fn = get_model_builder(name)
    return fn(**model_cfg)


def get_model_builder(name: str) -> Callable:
    """Get the model name and returns the model builder method.

    Args:
    ----
        name (str): The name under which the model is registered.

    """
    name = name.lower()
    if name not in BUILTIN_MODELS:
        raise ValueError(f"Unknown model {name}. Available models: {list_models()}")

    return BUILTIN_MODELS[name]


def list_models() -> list:
    """List all available models."""
    return list(BUILTIN_MODELS.keys())


def register_model(name: str | None = None) -> Callable:
    """Register a model.

    Args:
    ----
        name (str): The name of the model.

    """

    def decorator(model: torch.nn.Module) -> torch.nn.Module:
        BUILTIN_MODELS[name or model.__name__.lower()] = model
        return model

    return decorator
