import os
import pathlib
from typing import NewType, TypeVar

from PIL.Image import Image as PILImage

from src.utils.constants import DIFFUSERS_AVAILABLE, PANDAS_AVAILABLE, TRANSFORMERS_AVAILABLE

__all__ = ["DataFrame", "DiffusionPipeline", "PathLike", "PILImage", "TransformersPipeline"]

PathLike = TypeVar("PathLike", str, os.PathLike, pathlib.Path)

if DIFFUSERS_AVAILABLE:
    from diffusers import DiffusionPipeline as _DiffusionPipeline

    DiffusionPipeline = NewType("DiffusionPipeline", _DiffusionPipeline)
else:
    DiffusionPipeline = NewType("DiffusionPipeline", object)

if PANDAS_AVAILABLE:
    import pandas as pd

    DataFrame = NewType("DataFrame", pd.DataFrame)
else:
    DataFrame = NewType("DataFrame", object)

if TRANSFORMERS_AVAILABLE:
    from transformers import pipeline

    TransformersPipeline = NewType("TransformersPipeline", pipeline)
else:
    TransformersPipeline = NewType("TransformersPipeline", object)
