import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from src.utils.types import PILImage

__all__ = ["ToRGBTensor"]


class ToRGBTensor(T.ToTensor):
    """Convert a `PIL Image` or `numpy.ndarray` to tensor.

    Compared with the torchvision `ToTensor` transform, it converts images with a single channel to
    RGB images. In addition, the conversion to tensor is done only if the input is not already a
    tensor.
    """

    def __call__(self, pic: PILImage | np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert the input image to tensor.

        Args:
        ----
            pic (PIL PILImage | numpy.ndarray | torch.Tensor): Image to be converted to tensor.

        """
        img = F.to_tensor(pic) if isinstance(pic, (PILImage | np.ndarray)) else pic

        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        return img
