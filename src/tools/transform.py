from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import ImageDraw, ImageFont
from torchvision.io import decode_jpeg, encode_jpeg
from torchvision.transforms import InterpolationMode

from src.tools._api import register_tool
from src.tools._singletons import RMBG, StableDiffusionXLTurbo

__all__ = [
    "Transform",
    "AddGaussianNoise",
    "AddJPEGCompression",
    "ApplyCutMix",
    "ApplyMixUp",
    "ChangeBrightness",
    "ChangeContrast",
    "CropRandomShuffleAndRecompose",
    "DefocusBlurImage",
    "EditImageStyle",
    "EditImageWeather",
    "FlipImage",
    "Identity",
    "OverlayColor",
    "PasteGeneratedObjectAtRandomPosition",
    "PasteGeometricShapeAtRandomPosition",
    "PasteTextAtRandomPosition",
    "RotateImage",
    "ZoomAtRandomPosition",
]

_ORIENTATIONS = Literal["horizontal", "vertical"]
_SHAPES = Literal["circle", "square", "triangle"]


class Transform(ABC):
    """Base class for transformation tools."""

    @abstractmethod
    def __call__(self, sample: dict) -> dict:
        """Apply the transformation to the sample.

        Args:
        ----
            sample (dict): The sample to transform.

        """
        raise NotImplementedError


@register_tool(category="transform")
class AddGaussianNoise(Transform):
    """Add Gaussian noise to the input sample.

    Args:
    ----
        variance_factor (float): The factor to multiply the variance of the sample.
            Defaults to 1.4.

    Examples:
    --------
        Add Gaussian noise to the sample:
        >>> noise = AddGaussianNoise()
        >>> sample = {"images_tensor": torch.rand(3, 256, 256)}
        >>> sample_noise = noise(sample)

    """

    def __init__(self, variance_factor: float = 1.4) -> None:
        self.variance_factor = variance_factor

    def __call__(self, sample: dict) -> dict:
        """Apply the Gaussian noise transformation to the sample.

        Args:
        ----
            sample (dict): The sample to transform.

        """
        img = sample["images_tensor"]
        target_variance = img.var() * self.variance_factor
        noise = (target_variance - img.var()) ** 0.5 * torch.randn_like(img)
        sample["images_tensor"] = img + noise
        sample["images_pil"] = F.to_pil_image(sample["images_tensor"])

        return sample


@register_tool(category="transform")
class AddJPEGCompression(Transform):
    """Iteratively compress the sample until its peak signal-to-noise ratio reaches a target.

    Args:
    ----
        target_psnr (float): The target PSNR. Defaults to 26.0.

    Examples:
    --------
        Apply JPEG compression to the sample:
        >>> jpeg = AddJPEGCompression()
        >>> sample = {"images_tensor": torch.rand(3, 256, 256)}
        >>> sample_jpeg = jpeg(sample)

    """

    def __init__(self, target_psnr: float = 26.0) -> None:
        self.target_psnr = target_psnr

    def __call__(self, sample: dict) -> dict:
        """Apply the JPEG compression transformation to the sample.

        Args:
        ----
            sample (dict): The sample to transform.

        """
        img = sample["images_tensor"]
        original_shape = img.shape

        img = (img * 255).to(torch.uint8) if img.dtype == torch.float32 else img
        img = img.view((-1,) + img.shape[-3:])

        if img.shape[0] == 0:
            return img.reshape(original_shape).clone()

        img_jpeg = img
        channels = img_jpeg.shape[0]

        quality = 100
        psnr = float("inf")
        while psnr > self.target_psnr and quality > 0:
            img_jpeg = [decode_jpeg(encode_jpeg(img_jpeg[i], quality)) for i in range(channels)]
            img_jpeg_reshape = torch.stack(img_jpeg, dim=0).view(original_shape)
            psnr = get_image_psnr(img_jpeg_reshape / 255.0, img / 255.0)
            quality -= 2

        sample["images_tensor"] = img_jpeg_reshape / 255.0
        sample["images_pil"] = F.to_pil_image(sample["images_tensor"])

        return sample


@register_tool(category="transform")
class ApplyCutMix(Transform):
    """Paste on the input sample a random region of another sample.

    Args:
    ----
        alpha (float): Beta distribution parameter. Defaults to 1.0.

    Examples:
    --------
        Paste a random region of another sample on the sample:
        >>> cutmix = ApplyCutMix()
        >>> sample = {"_parent": src.data.ImageDataset(), "images_tensor": torch.rand(3, 256, 256)}
        >>> sample_cutmix = cutmix(sample)

    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def __call__(self, sample: dict) -> dict:
        """Apply the cutmix transformation to the sample.

        Args:
        ----
            sample (dict): The sample to transform.

        """
        lam = np.random.beta(self.alpha, self.alpha)

        img1 = sample["images_tensor"]

        parent = sample["_parent"]
        idx = torch.randint(0, len(parent), (1,)).item()
        img2 = parent[idx]["images_tensor"]

        # resize img2 to the same size as img1 while keeping the aspect ratio
        _, h, w = F.get_dimensions(img1)
        img2 = F.center_crop(F.resize(img2, max(h, w)), (h, w))

        # make a random bbox
        cut_ratio = np.sqrt(1 - lam)
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        bbx1, bby1, bbx2, bby2 = (
            np.clip(cx - cut_w // 2, 0, w),
            np.clip(cy - cut_h // 2, 0, h),
            np.clip(cx + cut_w // 2, 0, w),
            np.clip(cy + cut_h // 2, 0, h),
        )

        img1[:, bby1:bby2, bbx1:bbx2] = img2[:, bby1:bby2, bbx1:bbx2]
        lam = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (img1.shape[-1] * img1.shape[-2])

        sample["images_tensor"] = img1
        sample["images_pil"] = F.to_pil_image(sample["images_tensor"])

        return sample


@register_tool(category="transform")
class ApplyMixUp(Transform):
    """Mix the input sample with another sample randomly chosen from the dataset.

    Args:
    ----
        alpha (float): The mixing coefficient. Defaults to 0.7.

    Examples:
    --------
        Mix the sample with another sample:
        >>> mixup = ApplyMixUp()
        >>> sample = {"_parent": src.data.ImageDataset(), "images_tensor": torch.rand(3, 256, 256)}
        >>> sample_mixup = mixup(sample)

    """

    def __init__(self, alpha: float = 0.7) -> None:
        self.alpha = alpha

    def __call__(self, sample: dict) -> dict:
        """Apply the mixup transformation to the sample.

        Args:
        ----
            sample (dict): The sample to transform.

        """
        img1 = sample["images_tensor"]

        parent = sample["_parent"]
        idx = torch.randint(0, len(parent), (1,)).item()
        img2 = parent[idx]["images_tensor"]

        # resize img2 to the same size as img1 while keeping the aspect ratio
        _, h, w = F.get_dimensions(img1)
        img2 = F.center_crop(F.resize(img2, max(h, w)), (h, w))

        img_mixup = self.alpha * img1 + (1 - self.alpha) * img2

        sample["images_tensor"] = img_mixup
        sample["images_pil"] = F.to_pil_image(sample["images_tensor"])

        return sample


@register_tool(category="transform")
class ChangeBrightness(Transform):
    """Adjust the brightness of the input sample.

    Args:
    ----
        brightness_factor (float): How much to adjust the brightness. Can be any non-negative
            number. 0 gives a black image, 1 gives the original image while 2 increases the
            brightness by a factor of 2.

    Examples:
    --------
        Increase the brightness of the sample:
        >>> bright = ChangeBrightness(1.5)
        >>> sample = {"images_tensor": torch.rand(3, 256, 256)}
        >>> sample_bright = bright(sample)

        Decrease the brightness of the sample:
        >>> bright = ChangeBrightness(0.5)
        >>> sample = {"images_tensor": torch.rand(3, 256, 256)}
        >>> sample_bright = bright(sample)

    """

    def __init__(self, brightness_factor: float) -> None:
        self.brightness_factor = brightness_factor

    def __call__(self, sample: dict) -> dict:
        """Apply the brightness transformation to the sample.

        Args:
        ----
            sample (dict): The sample to transform.

        """
        img = sample["images_tensor"]
        img_bright = F.adjust_brightness(img, self.brightness_factor)

        sample["images_tensor"] = img_bright
        sample["images_pil"] = F.to_pil_image(sample["images_tensor"])

        return sample


@register_tool(category="transform")
class ChangeContrast(Transform):
    """Adjust the contrast of the input sample.

    Args:
    ----
        contrast_factor (float): How much to adjust the contrast. Can be any non-negative number.
            0 gives a solid gray image, 1 gives the original image while 2 increases the contrast
            by a factor of 2.

    Examples:
    --------
        Increase the contrast of the sample:
        >>> contrast = ChangeContrast(1.5)
        >>> sample = {"images_tensor": torch.rand(3, 256, 256)}
        >>> sample_contrast = contrast(sample)

        Decrease the contrast of the sample:
        >>> contrast = ChangeContrast(0.5)
        >>> sample = {"images_tensor": torch.rand(3, 256, 256)}
        >>> sample_contrast = contrast(sample)

    """

    def __init__(self, contrast_factor: float) -> None:
        self.contrast_factor = contrast_factor

    def __call__(self, sample: dict) -> dict:
        """Apply the contrast transformation to the sample.

        Args:
        ----
            sample (dict): The sample to transform.

        """
        img = sample["images_tensor"]
        img_contrast = F.adjust_contrast(img, self.contrast_factor)

        sample["images_tensor"] = img_contrast
        sample["images_pil"] = F.to_pil_image(sample["images_tensor"])

        return sample


@register_tool(category="transform")
class CropRandomShuffleAndRecompose(Transform):
    """Crop the sample into a grid of patches and randomly shuffle them spatially.

    Args:
    ----
        grid_size (int): The size of the grid. Defaults to 2.

    Examples:
    --------
        Crop the sample into a 3x3 grid and reshuffle the patches:
        >>> patch = CropRandomShuffleAndRecompose(3)
        >>> sample = {"images_tensor": torch.rand(3, 256, 256)}
        >>> img_patch = patch(img)

    """

    def __init__(self, grid_size: int = 2) -> None:
        self.grid_size = (grid_size, grid_size)
        self.size = (224, 224)

    def __call__(self, sample: dict) -> dict:
        """Apply the patch and reshuffle transformation to the sample.

        Args:
        ----
            sample (dict): The sample to transform.

        """
        img = sample["images_tensor"]
        grid_height, grid_width = self.grid_size
        cell_height, cell_width = img.shape[1] // grid_height, img.shape[2] // grid_width
        img_size = (cell_height * grid_height, cell_width * grid_width)
        resized_img = F.resize(img, size=img_size, antialias=True)

        # crop image and resize to output size
        crops = resized_img.unfold(1, cell_height, cell_height)
        crops = crops.unfold(2, cell_width, cell_width)
        crops = crops.reshape(3, -1, cell_height, cell_width).transpose(0, 1)
        crops = F.resize(crops, size=self.size, antialias=True)

        # shuffle patches
        indices = torch.randperm(crops.size(0))
        crops = crops[indices]

        # concatenate patches
        out_height = self.size[0] * grid_height
        out_width = self.size[1] * grid_width
        img = crops.reshape(grid_height, grid_width, 3, *self.size)
        img = img.permute(2, 0, 3, 1, 4).reshape(3, out_height, out_width)
        img = F.resize(img, size=(img_size[0], img_size[1]), antialias=True)

        sample["images_tensor"] = img
        sample["images_pil"] = F.to_pil_image(sample["images_tensor"])

        return sample


@register_tool(category="transform")
class DefocusBlurImage(Transform):
    """Blur the input sample using a Gaussian filter.

    Args:
    ----
        blur_factor (float): Estimate the target blur level as the initial sharpness level divided
            by the blur factor. Defaults to 10.0.

    Examples:
    --------
        Apply gaussian blur to the sample:
        >>> blur = DefocusBlurImage()
        >>> sample = {"images_tensor": torch.rand(3, 256, 256)}
        >>> sample_blur = blur(sample)

    """

    def __init__(self, blur_factor: float = 10.0, sigma: float = 1.0) -> None:
        self.blur_factor = blur_factor
        self.sigma = sigma

    def __call__(self, sample: dict) -> dict:
        """Apply the defocus blur transformation to the sample.

        Args:
        ----
            sample (dict): The sample to transform.

        """
        img = sample["images_tensor"]
        init_blur = get_image_sharpness(img)
        curr_blur = init_blur
        target_blur = init_blur / self.blur_factor

        blurred_img = img
        while curr_blur > target_blur:
            blurred_img = F.gaussian_blur(blurred_img, kernel_size=3, sigma=self.sigma)
            curr_blur = get_image_sharpness(blurred_img)

        sample["images_tensor"] = blurred_img
        sample["images_pil"] = F.to_pil_image(sample["images_tensor"])

        return sample


@register_tool(category="transform")
class EditImageStyle(Transform):
    """Regenerate an image with the input sample label and a specific style.

    Args:
    ----
        style (str): The visual style to apply to the sample.

    Examples:
    --------
        Generate an image given a label name in the style of a sculpture:
        >>> style = EditImageStyle("sculpture")
        >>> sample = {"labels_class_name": "cat"}
        >>> sample_style = style(sample)

        Generate an image given a label name in the style of a tattoo:
        >>> style = EditImageStyle("tattoo")
        >>> sample = {"labels_class_name": "dog"}
        >>> sample_style = style(sample)

    """

    def __init__(self, style: str) -> None:
        self.style = style
        self.model = StableDiffusionXLTurbo()
        self.prompt = "a %s of a %s"

    @torch.no_grad()
    def __call__(self, sample: dict) -> dict:
        """Apply the style transformation to the sample.

        Args:
        ----
            sample (dict): The sample to transform.

        """
        label = sample["labels_class_name"]

        prompt = self.prompt % (self.style, label)
        img_style_pil = self.model(prompt, guidance_scale=0.0, num_inference_steps=1).images[0]

        sample["images_tensor"] = F.to_tensor(img_style_pil)
        sample["images_pil"] = img_style_pil

        return sample


@register_tool(category="transform")
class EditImageWeather(Transform):
    """Regenerate an image with the input sample label and a specific weather.

    Args:
    ----
        weather (str): The weather to apply to the sample.

    Examples:
    --------
        Generate an image given a label name in a rainy weather:
        >>> weather = EditImageWeather("rainy")
        >>> sample = {"labels_class_name": "cat"}
        >>> sample_weather = weather(sample)

        Generate an image given a label name in a snowy weather:
        >>> weather = EditImageWeather("snowy")
        >>> sample = {"labels_class_name": "dog"}
        >>> sample_weather = weather(sample)

    """

    def __init__(self, weather: str) -> None:
        self.weather = weather
        self.model = StableDiffusionXLTurbo()
        self.prompt = "a photo of a %s in a %s weather"

    @torch.no_grad()
    def __call__(self, sample: dict) -> dict:
        """Apply the snow transformation to the sample.

        Args:
        ----
            sample (dict): The sample to transform.

        """
        label = sample["labels_class_name"]

        prompt = self.prompt % (label, self.weather)
        img_weather_pil = self.model(prompt, guidance_scale=0.0, num_inference_steps=1).images[0]

        sample["images_tensor"] = F.to_tensor(img_weather_pil)
        sample["images_pil"] = img_weather_pil

        return sample


@register_tool(category="transform")
class FlipImage(Transform):
    """Flip the input sample.

    Args:
    ----
        orientation ("horizontal" | "vertical"): The orientation of the flip.

    Examples:
    --------
        Flip the sample horizontally:
        >>> flip = FlipImage("horizontal")
        >>> sample = {"images_tensor": torch.rand(3, 256, 256)}
        >>> sample_flip = flip(sample)

        Flip the sample vertically:
        >>> flip = FlipImage("vertical")
        >>> sample = {"images_tensor": torch.rand(3, 256, 256)}
        >>> sample_flip = flip(sample)

    """

    def __init__(self, orientation: _ORIENTATIONS) -> None:
        self.orientation = orientation

    def __call__(self, sample: dict) -> dict:
        """Apply the flip transformation to the sample.

        Args:
        ----
            sample (dict): The sample to transform.

        """
        img = sample["images_tensor"]
        img_flip = F.hflip(img) if self.orientation == "horizontal" else F.vflip(img)

        sample["images_tensor"] = img_flip
        sample["images_pil"] = F.to_pil_image(sample["images_tensor"])

        return sample


@register_tool(category="transform")
class Identity(Transform):
    """Do not apply any transform and return the input sample.

    Args:
    ----
        None

    Examples:
    --------
        Apply the identity transformation to the sample:
        >>> identity = Identity()
        >>> sample = {"images_tensor": torch.rand(3, 256, 256)}
        >>> sample_identity = identity(sample)

    """

    def __call__(self, sample: dict) -> dict:
        """Apply the identity transformation to the sample.

        Args:
        ----
            sample (dict): The sample to transform.

        """
        return sample


@register_tool(category="transform")
class OverlayColor(Transform):
    """Overlay a color on the input sample.

    Args:
    ----
        color (tuple[int, int, int]): The RGB color to apply to the sample.
        opacity (float): The opacity of the color, between 0 and 1.

    Examples:
    --------
        Add a red color with 50% opacity to the sample:
        >>> color = OverlayColor((255, 0, 0), 0.5)
        >>> sample = {"images_tensor": torch.rand(3, 256, 256)}
        >>> sample_color = color(sample)

        Add a blue color with 100% opacity to the sample:
        >>> color = OverlayColor((0, 0, 255))
        >>> sample = {"images_tensor": torch.rand(3, 256, 256)}
        >>> sample_color = color(sample)

    """

    def __init__(self, color: tuple[int, int, int], opacity: float) -> None:
        self.color = torch.tensor(color, dtype=torch.float32) / 255.0
        self.opacity = opacity

    def __call__(self, sample: dict) -> dict:
        """Apply the color transformation to the sample.

        Args:
        ----
            sample (dict): The sample to transform.

        """
        img = sample["images_tensor"]
        color = torch.ones_like(img) * self.color.view(3, 1, 1)

        sample["images_tensor"] = (1 - self.opacity) * img + self.opacity * color
        sample["images_pil"] = F.to_pil_image(sample["images_tensor"])

        return sample


@register_tool(category="transform")
class PasteGeneratedObjectAtRandomPosition(Transform):
    """Paste a generated object on a random region of the input sample.

    Args:
    ----
        class_name (str | None): The name of the object to paste on the sample.
        size (int): The size of the object.
        repeat (int): The number of objects to paste.

    Examples:
    --------
        Paste one cat object on the sample:
        >>> paste_object = PasteGeneratedObjectAtRandomPosition("cat", 256, 1)
        >>> sample = {"images_tensor": torch.rand(3, 256, 256)}
        >>> sample_paste_object = paste_object(sample)

        Paste two dogs on the sample:
        >>> paste_object = PasteGeneratedObjectAtRandomPosition("dog", 256, 2)
        >>> sample = {"_parent": src.data.ImageDataset(), "images_tensor": torch.rand(3, 256, 256)}
        >>> sample_paste_object = paste_object(sample)

    """

    def __init__(self, class_name: str, size: int, repeat: int) -> None:
        self.class_name = class_name
        self.size = size
        self.repeat = repeat
        self.generate_model = StableDiffusionXLTurbo()
        self.background_remove_model = RMBG()

    @torch.no_grad()
    def __call__(self, sample: dict) -> dict:
        """Apply the paste shape transformation to the sample.

        Args:
        ----
            sample (dict): The sample to transform.

        """
        img = sample["images_tensor"]

        for _ in range(self.repeat):
            img = self._paste_object(img)

        sample["images_tensor"] = img
        sample["images_pil"] = F.to_pil_image(sample["images_tensor"])

        return sample

    def _paste_object(self, img: torch.Tensor) -> torch.Tensor:
        """Paste a shape on the input image.

        Args:
        ----
            img (torch.Tensor): The image to paste the shape on.

        """
        # generate an object with a flat background and remove the background
        prompt = "a photo of a %s, white background, full body" % self.class_name
        img2_pil = self.generate_model(prompt, guidance_scale=0.0, num_inference_steps=1).images[0]
        img2 = F.to_tensor(img2_pil)
        mask = F.to_tensor(self.background_remove_model(img2_pil, return_mask=True))

        # resize img2 to the required size
        _, h, w = F.get_dimensions(img)
        img2 = F.center_crop(F.resize(img2, self.size), (self.size, self.size))
        mask = F.center_crop(F.resize(mask, self.size), (self.size, self.size))

        # get a random location in the image
        x = torch.randint(0, w - self.size, (1,)).item()
        y = torch.randint(0, h - self.size, (1,)).item()

        # pad img2 to the size of img
        img2 = F.pad(img2, (x, y, w - x - self.size, h - y - self.size))
        mask = F.pad(mask, (x, y, w - x - self.size, h - y - self.size))

        # paste object on the image
        img = img * (1 - mask) + img2 * mask

        return img


@register_tool(category="transform")
class PasteGeometricShapeAtRandomPosition(Transform):
    """Paste a shape on a random region of the input sample.

    Args:
    ----
        shape ("circle", "square", "triangle"): The shape to paste on the sample.
        size (int): The size of the object.
        color (tuple[int, int, int]): The RGB color of the object.
        fill (bool): Whether to fill the object.
        repeat (int): The number of shapes to paste.

    Examples:
    --------
        Paste a green circle on the sample:
        >>> paste_shape = PasteGeometricShapeAtRandomPosition("circle", 48, (0, 255, 0), False, 1)
        >>> sample = {"images_tensor": torch.rand(3, 256, 256)}
        >>> sample_paste_shape = paste_shape(sample)

        Paste three red square on the sample:
        >>> paste_shape = PasteGeometricShapeAtRandomPosition("square", 48, (255, 0, 0), False, 3)
        >>> sample = {"images_tensor": torch.rand(3, 256, 256)}
        >>> sample_paste_shape = paste_shape(sample)

    """

    def __init__(
        self, shape: _SHAPES, size: int, color: tuple[int, int, int], fill: bool, repeat: int
    ) -> None:
        self.shape = shape
        self.size = size
        self.color = tuple(color)
        self.fill = fill
        self.repeat = repeat

    def __call__(self, sample: dict) -> dict:
        """Apply the paste shape transformation to the sample.

        Args:
        ----
            sample (dict): The sample to transform.

        """
        img = sample["images_tensor"]

        for _ in range(self.repeat):
            img = self._paste_shape(img)

        sample["images_tensor"] = img
        sample["images_pil"] = F.to_pil_image(sample["images_tensor"])

        return sample

    def _paste_shape(self, img: torch.Tensor) -> torch.Tensor:
        """Paste a shape on the input image.

        Args:
        ----
            img (torch.Tensor): The image to paste the shape on.

        """
        img_pil = F.to_pil_image(img)

        # get a random location in the image
        h, w = img.shape[1], img.shape[2]
        x = torch.randint(0, w - self.size, (1,)).item()
        y = torch.randint(0, h - self.size, (1,)).item()

        # paste shape on the image
        draw = ImageDraw.Draw(img_pil)
        if self.shape == "circle":
            draw.ellipse(
                (x, y, x + self.size, y + self.size),
                fill=self.color if self.fill else None,
                outline=self.color,
                width=8 if not self.fill else 1,
            )
        elif self.shape == "square":
            draw.rectangle(
                (x, y, x + self.size, y + self.size),
                fill=self.color if self.fill else None,
                outline=self.color,
                width=8 if not self.fill else 1,
            )
        elif self.shape == "triangle":
            draw.polygon(
                [(x, y), (x + self.size, y), (x + self.size // 2, y + self.size)],
                fill=self.color if self.fill else None,
                outline=self.color,
                width=8 if not self.fill else 1,
            )
        else:
            raise ValueError("Invalid shape %s" % self.shape)

        return F.to_tensor(img_pil)


@register_tool(category="transform")
class PasteTextAtRandomPosition(Transform):
    """Paste text on a random region of the input sample.

    Args:
    ----
        text (str): The text to paste on the sample.
        font_size (int): The font size of the text.
        font_color (tuple[int, int, int]): The RGB color of the text.
        repeat (int): The number of text to paste.

    Examples:
    --------
        Paste a blue "Hello, world!" text once on the sample:
        >>> paste_text = PasteTextAtRandomPosition("Hello, world!", 48, (0, 0, 255), 1)
        >>> sample = {"images_tensor": torch.rand(3, 256, 256)}
        >>> sample_paste_text = paste_text(sample)

    """

    def __init__(
        self,
        text: str,
        font_size: int,
        font_color: tuple[int, int, int],
        repeat: int,
    ) -> None:
        self.text = text
        self.font_size = font_size
        self.font_color = tuple(font_color)
        self.repeat = repeat

    def __call__(self, sample: dict) -> dict:
        """Apply the paste text transformation to the sample.

        Args:
        ----
            sample (dict): The sample to transform.

        """
        img = sample["images_tensor"]

        for _ in range(self.repeat):
            img = self._paste_text(img)

        sample["images_tensor"] = img
        sample["images_pil"] = F.to_pil_image(sample["images_tensor"])

        return sample

    def _paste_text(self, img: torch.Tensor) -> torch.Tensor:
        """Paste text on the input image.

        Args:
        ----
            img (torch.Tensor): The image to paste the text on.

        """
        img_pil = F.to_pil_image(img)

        # estimate the dimensions of the text
        font = ImageFont.truetype("arial.ttf", self.font_size)
        text_width, text_height = int(font.getlength(self.text)), self.font_size

        # get a random location in the image in the left part
        h, w = img.shape[1], img.shape[2]
        x = torch.randint(0, w - text_width, (1,)).item()
        y = torch.randint(0, h - text_height, (1,)).item()

        # paste text on the image
        draw = ImageDraw.Draw(img_pil)
        draw.text((x, y), self.text, fill=self.font_color, font_size=self.font_size)

        return F.to_tensor(img_pil)


@register_tool(category="transform")
class RotateImage(Transform):
    """Rotate the input sample.

    Args:
    ----
        angle (int): The angle of rotation.

    Examples:
    --------
        Rotate the sample to the right:
        >>> rotate = RotateImage(90)
        >>> sample = {"images_tensor": torch.rand(3, 256, 256)}
        >>> sample_rotate = rotate(sample)

        Rotate the sample to the left:
        >>> rotate = RotateImage(-90)
        >>> sample = {"images_tensor": torch.rand(3, 256, 256)}
        >>> sample_rotate = rotate(sample)

    """

    def __init__(self, angle: int) -> None:
        self.angle = angle

    def __call__(self, sample: dict) -> dict:
        """Apply the rotation transformation to the sample.

        Args:
        ----
            sample (dict): The sample to transform.

        """
        img = sample["images_tensor"]
        img_rotate = F.rotate(
            img, self.angle, expand=True, interpolation=InterpolationMode.BILINEAR
        )

        sample["images_tensor"] = img_rotate
        sample["images_pil"] = F.to_pil_image(sample["images_tensor"])

        return sample


@register_tool(category="transform")
class ZoomAtRandomPosition(Transform):
    """Zoom on a random region of the input sample.

    Args:
    ----
        zoom_factor (float): The zoom factor. Defaults to 2.0.

    Examples:
    --------
        Zoom on a random region of the sample:
        >>> zoom = ZoomAtRandomPosition()
        >>> sample = {"images_tensor": torch.rand(3, 256, 256)}
        >>> sample_zoom = zoom(sample)

    """

    def __init__(self, zoom_factor: int = 2) -> None:
        self.zoom_factor = zoom_factor
        self.size = (224, 224)

    def __call__(self, sample: dict) -> dict:
        """Apply the zoom transformation to the sample.

        Args:
        ----
            sample (dict): The sample to transform.

        """
        img = sample["images_tensor"]
        _, h, w = F.get_dimensions(img)
        th, tw = self.size

        if h < th or w < tw:
            raise ValueError(
                f"Required crop size {(th, tw)} is larger than input image size {(h, w)}"
            )

        if w == tw and h == th:
            i, j, th, tw = 0, 0, h, w
        else:
            i = torch.randint(0, h - th + 1, size=(1,)).item()
            j = torch.randint(0, w - tw + 1, size=(1,)).item()

        crop = F.crop(img, i, j, th, tw)
        zoom = F.resize(crop, (int(th * self.zoom_factor), int(tw * self.zoom_factor)))

        sample["images_tensor"] = zoom
        sample["images_pil"] = F.to_pil_image(sample["images_tensor"])

        return sample


def get_image_sharpness(img: torch.Tensor) -> float:
    """Estimate the image sharpness using the variance of the Laplacian.

    Higher values indicate a sharper image, while lower values indicate a blurrier image.

    Args:
    ----
        img (torch.Tensor): The image to estimate the blur level.

    """
    if img.shape[0] == 3:
        img = F.rgb_to_grayscale(img)

    if img.max() <= 1:
        img = img * 255

    laplacian_filter = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32)

    # Apply Laplacian filter
    laplacian = torch.nn.functional.conv2d(img, laplacian_filter)

    # Compute variance of Laplacian
    laplacian_variance = torch.var(laplacian)

    return laplacian_variance.item()


def get_image_psnr(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Compute the peak signal-to-noise ratio (PSNR) between two images.

    Args:
    ----
        img1 (torch.Tensor): The first image.
        img2 (torch.Tensor): The second image.

    """
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
