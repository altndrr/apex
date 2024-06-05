import gc
import sys

import torch

from src import utils
from src.utils._console_utils import BAR_FORMAT
from src.utils.types import DiffusionPipeline, TransformersPipeline

__all__ = [
    "reset_singletons",
    "InstructPix2Pix",
    "RMBG",
    "StableDiffusionXL",
    "StableDiffusionXLTurbo",
    "SentenceBERT",
]

_PROGRESS_BAR_CONFIG = {
    "desc": "Processing",
    "disable": False,
    "leave": False,
    "dynamic_ncols": True,
    "file": sys.stdout,
    "smoothing": 0,
    "bar_format": BAR_FORMAT,
}


def reset_singletons() -> None:
    """Reset the singletons.

    Useful to free up memory and GPU resources.
    """
    del InstructPix2Pix._instance
    del RMBG._instance
    del StableDiffusionXL._instance
    del StableDiffusionXLTurbo._instance
    del SentenceBERT._instance

    InstructPix2Pix._instance = None
    RMBG._instance = None
    StableDiffusionXL._instance = None
    StableDiffusionXLTurbo._instance = None
    SentenceBERT._instance = None

    gc.collect()
    torch.cuda.empty_cache()


class InstructPix2Pix:
    """Singleton of the InstructPix2Pix model for image editing.

    Args:
    ----
        model_name (str): Name of the model to use. Defaults to "timbrooks/instruct-pix2pix".

    """

    _instance = None

    def __new__(cls, model_name: str = "timbrooks/instruct-pix2pix") -> DiffusionPipeline:
        assert utils.DIFFUSERS_AVAILABLE, "diffusers is not available."
        from diffusers import (
            EulerAncestralDiscreteScheduler,
            StableDiffusionInstructPix2PixPipeline,
        )

        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                model_name, torch_dtype=torch.float16, safety_checker=None, device_map="auto"
            )
            cls._instance.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                cls._instance.pipe.scheduler.config
            )
            cls._instance.pipe.set_progress_bar_config(**_PROGRESS_BAR_CONFIG)
        return cls._instance.pipe


class RMBG:
    """Singleton of the RMBG model for background removal.

    Args:
    ----
        model_name (str): Name of the model to use. Defaults to "briaai/RMBG-1.4".

    """

    _instance = None

    def __new__(cls, model_name: str = "briaai/RMBG-1.4") -> TransformersPipeline:
        assert utils.TRANSFORMERS_AVAILABLE, "transformers is not available."
        from transformers import pipeline

        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.pipe = pipeline(
                "image-segmentation", model=model_name, trust_remote_code=True
            )
        return cls._instance.pipe


class StableDiffusionXL:
    """Singleton of the StableDiffusionXL model for image generation.

    Args:
    ----
        model_name (str): Name of the model to use. Defaults to
            "stabilityai/stable-diffusion-xl-base-1.0".

    """

    _instance = None

    def __new__(
        cls, model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"
    ) -> DiffusionPipeline:
        assert utils.DIFFUSERS_AVAILABLE, "diffusers is not available."
        from diffusers import AutoPipelineForText2Image

        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.pipe = AutoPipelineForText2Image.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
                device_map="auto",
            )
            cls._instance.pipe.set_progress_bar_config(**_PROGRESS_BAR_CONFIG)
        return cls._instance.pipe


class StableDiffusionXLTurbo:
    """Singleton of the StableDiffusionXLTurbo model for image generation.

    Args:
    ----
        model_name (str): Name of the model to use. Defaults to "stabilityai/sdxl-turbo".

    """

    _instance = None

    def __new__(cls, model_name: str = "stabilityai/sdxl-turbo") -> DiffusionPipeline:
        assert utils.DIFFUSERS_AVAILABLE, "diffusers is not available."
        from diffusers import AutoPipelineForText2Image

        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.pipe = AutoPipelineForText2Image.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
                device_map="auto",
            )
            cls._instance.pipe.set_progress_bar_config(**_PROGRESS_BAR_CONFIG)

        return cls._instance.pipe


class SentenceBERT:
    """Singleton of SentenceBERT for sentence embeddings.

    Attributes
    ----------
        model_name (str): Name of the model to use. Defaults to
            "sentence-transformers/all-MiniLM-L6-v2".

    """

    _instance = None

    def __new__(cls, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> "SentenceBERT":
        assert utils.TRANSFORMERS_AVAILABLE, "transformers is not available."
        from transformers import AutoModel, AutoTokenizer

        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registered_names = []
            cls._instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cls._instance.model = AutoModel.from_pretrained(model_name)
            cls._instance.tokenizer = AutoTokenizer.from_pretrained(model_name)
            cls._instance.model = cls._instance.model.to(cls._instance.device)
        return cls._instance

    def __call__(self, sentence: list[str]) -> torch.Tensor:
        """Encode the input sentence with the BERT model.

        Args:
        ----
            sentence (list[str]): Input sentences.

        """
        assert self.tokenizer is not None

        tokens = self.tokenizer(
            sentence, max_length=128, truncation=True, padding="max_length", return_tensors="pt"
        ).to(self.device)
        embeddings = self.model(**tokens).last_hidden_state

        # mask out padding tokens
        mask = tokens["attention_mask"].unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask

        # sum over all tokens
        summed = torch.sum(masked_embeddings, dim=1)
        summed_mask = torch.clamp(mask.sum(dim=1), min=1e-9)

        # normalize and remove batch dimension
        embeddings = summed / summed_mask
        embeddings = embeddings.squeeze(0)

        return embeddings

    def __getattr__(self, name: str) -> torch.Tensor:
        """Get the attribute from the model.

        Args:
        ----
            name (str): Name of the attribute.

        """
        if name not in self._registered_names:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

        return getattr(self.model, name)

    def register_buffer(self, name: str, buffer: torch.Tensor, exists_ok: bool = False) -> None:
        """Register a buffer with the model.

        Args:
        ----
            name (str): Name of the buffer.
            buffer (torch.Tensor): Buffer to register.
            exists_ok (bool, optional): Whether to allow overwriting existing buffers.
                Defaults to False.

        """
        if hasattr(self.model, name):
            if not exists_ok:
                raise ValueError(f"Buffer with name '{name}' already exists.")
            return

        self._registered_names.append(name)
        self.model.register_buffer(name, buffer)
