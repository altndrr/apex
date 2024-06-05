from abc import ABC, abstractmethod
from pathlib import Path
from time import time

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor

from src.data import ImageDataset
from src.tools._api import register_tool
from src.tools._singletons import SentenceBERT, StableDiffusionXLTurbo
from src.utils import get_logger

log = get_logger(__name__, rank_zero_only=True)


__all__ = ["TextToImageGeneration", "TextToImageRetrieval"]


class Select(ABC):
    """Base class for selection tools."""

    @abstractmethod
    def __call__(self, dataset: ImageDataset) -> dict:
        """Apply the selection tool.

        Args:
        ----
            dataset (ImageDataset): The dataset to (optionally) select from.

        """
        raise NotImplementedError


@register_tool(category="select")
class TextToImageGeneration(Select):
    """Generate an image with a class and image type.

    Args:
    ----
        class_name (str | "random"): The class name of the object to generate. If "random", the
            class name is randomly selected from the dataset.
        image_type (str): The type of image. Default to "photo".

    Examples:
    --------
        Generate an oil painting of a dog:
        >>> generate_dog = TextToImageGeneration("dog", "oil painting")
        >>> dataset = ...
        >>> sample_generation = generate_dog(sample)

        Generate a pencil sketch of a labrador:
        >>> generate_dog = TextToImageGeneration("labrador", "pencil sketch")
        >>> dataset = ...
        >>> sample_generation = generate_dog(sample)

    """

    def __init__(self, class_name: str, image_type: str = "photo") -> None:
        if class_name and "random" in class_name:
            class_name = None

        self.model = StableDiffusionXLTurbo()
        self.image_type = image_type
        self.class_name = class_name

    @torch.no_grad()
    def __call__(self, dataset: ImageDataset) -> dict:
        """Apply the generate image selection.

        Args:
        ----
            dataset (ImageDataset): The dataset to (optionally) select from.

        """
        class_name = self.class_name
        if not class_name:
            idx = torch.randint(0, len(dataset.class_names), (1,)).item()
            class_name = dataset.class_names[idx]

        prompt = f"a {self.image_type} of a {class_name}"
        img_pil = self.model(prompt, guidance_scale=0.0, num_inference_steps=1).images[0]
        img_tensor = to_tensor(img_pil)

        sample = {}
        folder_name = class_name.replace(" ", "_")
        sample["_parent"] = dataset
        sample["images_fp"] = f".cache/data/generated/val/{folder_name}/{time()}.jpg"
        sample["images_tensor"] = img_tensor
        sample["images_pil"] = img_pil
        sample["labels_class_name"] = class_name

        # save image to disk
        Path(sample["images_fp"]).parent.mkdir(parents=True, exist_ok=True)
        img_pil.save(sample["images_fp"])

        return sample


@register_tool(category="select")
class TextToImageRetrieval(Select):
    """Retrieve an image from a dataset with a class and an image type.

    If the class name or the image type are not defined for the dataset, retrieval is replaced
    by generation.

    Args:
    ----
        class_name (str | "random"): The class name of the object to generate. If "random", the
            class name is randomly selected from the dataset.
        image_type (str): The type of image. Default to "photo".

    Examples:
    --------
        Retrieve an image of a random class name:
        >>> retrieve_random = TextToImageRetrieval("random")
        >>> dataset = ...
        >>> sample_selection = retrieve_random(dataset)

        Retrieve an image of a siamese cat:
        >>> retrieve_cat = TextToImageRetrieval("siamese cat")
        >>> dataset = ...
        >>> sample_selection = retrieve_cat(dataset)

    """

    def __init__(self, class_name: str, image_type: str = "photo") -> None:
        if class_name and "random" in class_name:
            class_name = None

        self.sentence_bert = SentenceBERT()
        self.class_name = class_name
        self.image_type = image_type

        self._alt_generation = TextToImageGeneration(self.class_name, self.image_type)
        self._sample_idxs = None
        self._prev_selected_sample_idxs = set()
        self._use_alt_generation = False

    @torch.no_grad()
    def __call__(self, dataset: ImageDataset) -> dict:
        """Apply the selection to the dataset.

        Args:
        ----
            dataset (ImageDataset): The dataset to (optionally) select from.

        """
        if self._use_alt_generation:
            return self._alt_generation(dataset)

        if self.image_type not in dataset._AVAILABLE_IMAGE_TYPES:
            if not self._use_alt_generation:
                log.warning("Type <%s> not found in dataset. Using generation...", self.image_type)
                self._use_alt_generation = True
            return self._alt_generation(dataset)

        class_name = self.class_name
        if class_name:
            # encode the class names
            if not hasattr(self.sentence_bert, "class_names_z"):
                class_names_z = F.normalize(self.sentence_bert(dataset.class_names), dim=-1)
                self.sentence_bert.register_buffer("class_names_z", class_names_z)

            # retrieve the closest class name
            if self._sample_idxs is None:
                query_z = F.normalize(self.sentence_bert([self.class_name]), dim=-1)
                similarities = self.sentence_bert.class_names_z @ query_z
                closest_idx = similarities.argmax().item()
                closest_name = dataset.class_names[closest_idx]

                sample_idx = torch.where(torch.tensor(dataset.labels_class_idx) == closest_idx)[0]
                self._sample_idxs = set(sample_idx.tolist())

                closest_similarity = similarities[closest_idx].item()
                if closest_similarity < 0.95:
                    if not self._use_alt_generation:
                        log.warning(
                            "Closest class name <%s> (%.2f). Using generation...",
                            closest_name,
                            closest_similarity,
                        )
                        self._use_alt_generation = True
                    return self._alt_generation(dataset)

                self.class_name = closest_name

            # sample an image with the class name
            sample_idxs = list(self._sample_idxs - self._prev_selected_sample_idxs)
            if len(sample_idxs) == 0:
                log.warning("Exhausted class <%s>. Start sampling with replacement.", class_name)
                sample_idxs = self._sample_idxs
                self._prev_selected_sample_idxs = set()
                sample_idxs = list(sample_idxs - self._prev_selected_sample_idxs)

            idx = sample_idxs[torch.randint(0, len(sample_idxs), (1,)).item()]
            self._prev_selected_sample_idxs.add(idx)
        elif not class_name:
            # sample an image with a random class name
            idx = torch.randint(0, len(dataset), (1,)).item()

        return dataset[idx]
