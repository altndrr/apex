from src.utils._pkg_resources_utils import package_available

__all__ = [
    "ACCELERATE_AVAILABLE",
    "BITSANDBYTES_AVAILABLE",
    "DIFFUSERS_AVAILABLE",
    "IMAGENET_X_AVAILABLE",
    "KAGGLE_AVAILABLE",
    "PANDAS_AVAILABLE",
    "SENTENCEPIECE_AVAILABLE",
    "TRANSFORMERS_AVAILABLE",
    "WANDB_AVAILABLE",
]

ACCELERATE_AVAILABLE = package_available("accelerate")
BITSANDBYTES_AVAILABLE = package_available("bitsandbytes")
DIFFUSERS_AVAILABLE = package_available("diffusers")
IMAGENET_X_AVAILABLE = package_available("imagenet_x")
KAGGLE_AVAILABLE = package_available("kaggle")
LOVELY_TENSORS_AVAILABLE = package_available("lovely-tensors")
PANDAS_AVAILABLE = package_available("pandas")
RICH_AVAILABLE = package_available("rich")
SENTENCEPIECE_AVAILABLE = package_available("sentencepiece")
TRANSFORMERS_AVAILABLE = package_available("transformers")
WANDB_AVAILABLE = package_available("wandb")
