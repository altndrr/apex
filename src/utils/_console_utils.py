import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal

import rich
import rich.syntax
import rich.tree
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm

from src.utils import _logging_utils
from src.utils.decorators import rank_zero_only

__all__ = ["print_config_tree"]

log = _logging_utils.get_logger(__name__, rank_zero_only=True)

BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}{postfix}]"


def format_iterable(
    iterable: Iterable,
    candidates: torch.Tensor | Mapping[str, torch.Tensor | float | int],
    prefix: str | None = None,
) -> None:
    """Add values as postfix string to progressbar.

    Args:
    ----
        iterable (Iterable): Progress bar (on global rank zero) or iterable (every other rank).
        candidates (torch.Tensor | Mapping[str, torch.Tensor | float | int]): The values
            to add as postfix strings to the progressbar.
        prefix (str, optional): The prefix to add to each of these values. Defaults to None.

    """
    if not isinstance(iterable, tqdm):
        raise ValueError("iterable type not supported")

    postfix_str = ""
    if isinstance(candidates, torch.Tensor):
        float_candidates = candidates.item()
        postfix_str += (
            f" {prefix}/loss: {float_candidates:.3f}"
            if prefix
            else f" loss: {float_candidates:.3f}"
        )
    elif isinstance(candidates, Mapping):
        for k, v in candidates.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if not isinstance(v, (float | int)):
                continue
            postfix_str += f" {prefix}/{k}: {v:.3f}" if prefix else f" {k}: {v:.3f}"

    if postfix_str:
        iterable.set_postfix_str(postfix_str)


def get_iterable(iterable: Iterable, name: Literal["tqdm"] = "tqdm", **kwargs) -> tqdm:
    """Return a progress bar.

    Args:
    ----
        iterable (Iterable): The iterable to wrap with a progress bar.
        name ("tqdm"): The name of the progress bar to return. Defaults to "tqdm".
        kwargs: Keyword arguments to pass to the progress bar.

    """
    if name != "tqdm":
        raise ValueError(f"Progress bar name {name} not supported")

    position = 2 * kwargs.pop("position") if "position" in kwargs else None

    return tqdm(
        iterable,
        desc=kwargs.pop("desc", "Processing"),
        position=position,
        disable=kwargs.pop("disable", False),
        leave=kwargs.pop("leave", True),
        dynamic_ncols=kwargs.pop("dynamic_ncols", True),
        file=kwargs.pop("file", sys.stdout),
        smoothing=kwargs.pop("smoothing", 0),
        bar_format=kwargs.pop("bar_format", BAR_FORMAT),
        **kwargs,
    )


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = ("data", "logger", "model", "task"),
    ignore: Sequence[str] = ("extras", "paths", "name", "tags"),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Print content of DictConfig using Rich library and its tree structure.

    Args:
    ----
        cfg (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines the printing order of config components.
        ignore (Sequence[str], optional): Determines the fields to skip when printing the config.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
        save_to_file (bool, optional): Whether to export config to the hydra output folder.

    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        if field in cfg:
            queue.append(field)

    # add all the other fields to queue (except those in `do_not_print` list)
    for field in cfg:
        if field not in queue and field not in ignore:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, (DictConfig | ListConfig)):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)
