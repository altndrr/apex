import logging
import os
import warnings

import torch
from omegaconf import DictConfig

from src.utils import _logging_utils
from src.utils._console_utils import print_config_tree
from src.utils._pkg_resources_utils import package_available
from src.utils.decorators import rank_zero_only

__all__ = ["extras"]

log = _logging_utils.get_logger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Apply optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Rich config printing
    - Monkey-patching tensor classes to have pretty representations
    - Setting precision of float32 matrix multiplication

    Args:
    ----
        cfg (DictConfig): Configuration composed by Hydra.

    """
    # set the rank of the process
    rank_zero_only.rank = int(os.environ.get("LOCAL_RANK", 0))

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable specific loggers
    if cfg.extras.get("disable_loggers"):
        disable_loggers = cfg.extras.disable_loggers
        log.info("Ignoring loggers! <cfg.extras.disable_loggers=%s>", disable_loggers)
        for disable_logger in disable_loggers:
            logging.getLogger(disable_logger).setLevel(logging.ERROR)

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # monkey-patch tensor classes to have pretty representations
    if cfg.extras.get("lovely_tensors"):
        assert package_available("lovely-tensors"), "lovely-tensors package not installed."

        import lovely_tensors as lt

        log.info("Applying monkey-patch for lovely-tensors! <cfg.extras.lovely_tensors=True>")
        lt.monkey_patch()

    # set precision of float32 matrix multiplication
    if cfg.extras.get("matmul_precision"):
        matmul_precision = cfg.extras.matmul_precision
        log.info(
            "Setting precision of matrix multiplication! <cfg.extras.matmul_precision=%s>",
            matmul_precision,
        )
        torch.set_float32_matmul_precision(matmul_precision)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        print_config_tree(cfg, resolve=True, save_to_file=True)
