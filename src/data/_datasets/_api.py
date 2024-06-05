from collections.abc import Callable
from pathlib import Path

from omegaconf import DictConfig
from torch.utils.data import Dataset

from src.utils.types import PathLike

__all__ = [
    "BUILTIN_DATASETS",
    "get_dataset",
    "get_dataset_builder",
    "list_datasets",
    "register_dataset",
]

BUILTIN_DATASETS = {}


def get_dataset(
    name: str,
    data_dir: PathLike = "data",
    split: str | None = None,
    download: bool = False,
    **data_cfg: DictConfig | dict,
) -> Dataset:
    """Get a dataset.

    Args:
    ----
        name (str): The name of the dataset.
        data_dir (PathLike): The directory where the dataset will be downloaded.
            Defaults to "data".
        split (str, optional): The split of the dataset to use. Defaults to None.
        download (bool): Whether to download the dataset. Defaults to False.
        data_cfg (DictConfig | dict): Additional arguments to pass to the dataset.

    """
    fn = get_dataset_builder(name)

    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)

    return fn(root=str(data_dir / name), split=split, download=download, **data_cfg)


def get_dataset_builder(name: str) -> Callable:
    """Get the builder of a dataset.

    Args:
    ----
        name (str): The name of the dataset.

    """
    name = name.lower()
    if name not in BUILTIN_DATASETS:
        raise ValueError(f"Unknown dataset {name}. Available datasets: {list_datasets()}")

    return BUILTIN_DATASETS[name]


def list_datasets() -> list:
    """List all available datasets."""
    return list(BUILTIN_DATASETS.keys())


def register_dataset(name: str | None = None) -> Callable:
    """Register a dataset.

    Args:
    ----
        name (str, optional): The name of the dataset.

    """

    def decorator(dataset: Dataset) -> Dataset:
        BUILTIN_DATASETS[name or dataset.__name__.lower()] = dataset
        return dataset

    return decorator
