from collections.abc import Sequence

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data._datasets import ImageDataset

__all__ = ["default_collate_fn", "get_dataloader"]


def default_collate_fn(batch: Sequence[dict]) -> dict:
    """Collate function for the dataloader.

    Args:
    ----
        batch (Sequence[dict]): List of samples.

    """
    if not isinstance(batch[0], dict):
        raise ValueError("Collate function expects a list of dictionaries")

    collated = {}
    for key in batch[0]:
        data_type = type(batch[0][key])
        values = [batch[i][key] for i in range(len(batch))]
        if data_type == torch.Tensor and all(values[0].shape == value.shape for value in values):
            collated[key] = torch.stack([item[key] for item in batch])
        elif data_type == dict:
            inner_collate = default_collate_fn([item[key] for item in batch])
            for inner_key in inner_collate:
                if inner_key in collated:
                    raise ValueError(f"Duplicate key: {inner_key}")
                collated[inner_key] = inner_collate[inner_key]
        else:
            collated[key] = [item[key] for item in batch]

    return collated


def get_dataloader(dataset: ImageDataset, data_cfg: DictConfig | dict) -> DataLoader:
    """Get a dataloader for a dataset.

    Args:
    ----
        dataset (ImageDataset): The dataset to use.
        data_cfg (DictConfig | dict): The configuration for the dataloader.

    """
    return DataLoader(
        dataset,
        batch_size=data_cfg.get("batch_size", 64),
        shuffle=data_cfg.get("shuffle", False),
        pin_memory=data_cfg.get("pin_memory", torch.cuda.is_available()),
        num_workers=data_cfg.get("num_workers", 4),
        drop_last=data_cfg.get("drop_last", False),
        collate_fn=data_cfg.get("collate_fn", default_collate_fn),
    )
