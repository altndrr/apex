import os
from collections.abc import Mapping
from typing import Any, Literal, NewType

from omegaconf import DictConfig, OmegaConf

from src import utils

log = utils.get_logger(__name__, rank_zero_only=True)

__all__ = ["Logger", "close_loggers", "get_logger", "log_hyperparameters"]

Logger = NewType("Logger", Any)


@utils.rank_zero_only
def close_loggers() -> None:
    """Ensure all loggers are closed properly."""
    if utils.WANDB_AVAILABLE:
        import wandb

        if wandb.run:
            wandb.finish()


def get_logger(name: Literal["wandb"], **logger_cfg: DictConfig | dict) -> Logger:
    """Get a logger.

    Args:
    ----
        name ("wandb"): Name of the logger.
        logger_cfg (dict): Configuration of the logger.

    """
    if name == "wandb":
        assert utils.WANDB_AVAILABLE, "wandb is not available"
        import wandb

        if wandb.run is not None:
            return wandb.run

        offline = logger_cfg.pop("offline", False)
        if offline:
            os.environ["WANDB_MODE"] = "dryrun"

        project = logger_cfg.pop("project") or os.getenv("WANDB_PROJECT", "apex")
        entity = logger_cfg.pop("entity") or os.getenv("WANDB_ENTITY", None)
        dir = logger_cfg.pop("save_dir") or logger_cfg.pop("dir")
        run = wandb.init(project=project, entity=entity, dir=dir, **logger_cfg)

        return run
    else:
        raise ValueError(f"Unknown logger {name}")


@utils.rank_zero_only
def log_hyperparameters(logger: Logger, object_dict: Mapping[str, Any]) -> None:
    """Log the hyperparams to the logger.

    Args:
    ----
        logger (Logger): List of loggers to log the hyperparams.
        object_dict (dict): Dictionary of objects to log.

    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict.get("model")

    hparams["model"] = cfg["model"]

    # save number of model parameters
    if model is not None:
        hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
        hparams["model/params/trainable"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        hparams["model/params/non_trainable"] = sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        )

    hparams["data"] = cfg["data"]
    hparams["extras"] = cfg.get("extras")

    hparams["seed"] = cfg.get("seed")

    # send hparams to the logger
    is_valid_logger_type = False
    if utils.WANDB_AVAILABLE:
        from wandb.wandb_run import Run

        if isinstance(logger, Run):
            logger.config.update(hparams, allow_val_change=True)
            is_valid_logger_type = True

    if not is_valid_logger_type:
        log.warning("Unrecognized logger type. Hyperparameters not logged.")
