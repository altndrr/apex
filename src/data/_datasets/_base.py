from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any, Literal

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import Compose
from torchvision.transforms.functional import to_tensor

from src import utils
from src.utils.types import PathLike, PILImage

log = utils.get_logger(__name__, rank_zero_only=True)

__all__ = ["ImageDataset", "ExperimentDataset"]


class ImageDataset(VisionDataset):
    """Dataset of images.

    Labels can be registered to the dataset using the `register_labels` method. By default,
    no labels are registered.

    Args:
    ----
        root (PathLike): Root directory of dataset where `images` are found.
        split (str, optional): The split of the dataset to use. Defaults to None.
        images (Sequence[str], optional): List of images. Defaults to None.
        download (bool): Whether to download the dataset. Defaults to False.
        kwargs: Extra arguments to pass to the dataset or dataloader.

    """

    _AVAILABLE_SPLITS: Sequence[str]
    _AVAILABLE_IMAGE_TYPES: Sequence[str]
    _BUILTIN_LABELS: set

    def __init__(
        self,
        root: PathLike,
        split: str | None = None,
        images: Sequence[str] | None = None,
        download: bool = False,
        **kwargs,
    ) -> None:
        self._BUILTIN_LABELS = set()

        self.name = Path(root).name

        if download:
            self.download(root)

        if split == "test" and split not in self._AVAILABLE_SPLITS:
            split = "val"
            log.warning("Split 'test' not available. Using 'val' split instead.")

        if split is not None and split not in self._AVAILABLE_SPLITS:
            raise ValueError(f"Split {split} not available for {self.__class__.__name__}")

        root = str(Path(root) / split) if split is not None else root

        if not images:
            images = [str(path) for path in Path(root).glob("*/*")]

        self.images = images

        self.root = root
        self.split = split

        paired_image_loader_kwargs = kwargs.get("paired_image_loader", {})
        self.loader = _default_image_loader
        self.pairs_loader = partial(_default_paired_image_loader, **paired_image_loader_kwargs)

        self.kwargs = kwargs

    @staticmethod
    def download(root: PathLike = "", exist_ok: bool = True) -> None:
        """Download the dataset.

        Args:
        ----
            root (PathLike): The output directory where the dataset will be downloaded.
            exist_ok (bool): Whether to raise an error if the output directory
                already exists. Defaults to True.

        """
        return

    def filter(self, mask: Sequence[bool], in_place: bool = False) -> "ImageDataset":
        """Filter the dataset.

        Args:
        ----
            mask (Sequence[bool]): The mask to filter the dataset.
            in_place (bool): Whether to filter the dataset in place. Defaults to False.

        """
        images = [image for image, keep in zip(self.images, mask, strict=True) if keep]

        to_register = {}
        for label in sorted(self._BUILTIN_LABELS):
            labels_idxs = getattr(self, f"labels_{label}_idx")
            labels_idxs = [label for label, keep in zip(labels_idxs, mask, strict=True) if keep]
            labels_names = getattr(self, f"{label}_names")
            to_register[label] = (labels_names, labels_idxs)

        if in_place:
            self.images = images
            for label, (names, indices) in to_register.items():
                setattr(self, f"{label}_names", names)
                setattr(self, f"{label}_to_idx", {c: i for i, c in enumerate(names)})
                setattr(self, f"labels_{label}_idx", indices)
            return self

        filtered = ImageDataset(root=self.root, split=self.split, images=images, **self.kwargs)
        for label, (names, indices) in to_register.items():
            filtered.register_labels(label, names, indices)
        return filtered

    def register_labels(
        self, label_name: str, names: Sequence[str], indices: Sequence[int]
    ) -> None:
        """Register a label to the dataset.

        Args:
        ----
            label_name (str): The name of the label.
            names (Sequence[str]): List of class names.
            indices (Sequence[int]): List of idx labels.

        """
        if label_name in self._BUILTIN_LABELS:
            raise ValueError(f"Label {label_name} is already registered in the dataset")

        label_names_attr = f"{label_name}_names"
        label_name_to_idx_attr = f"{label_name}_to_idx"
        label_idxs_attr = f"labels_{label_name}_idx"

        if label_names_attr in self.__dict__:
            raise ValueError(f"Label {label_name} already exists in the dataset")
        if label_name_to_idx_attr in self.__dict__:
            raise ValueError(f"Label {label_name} already exists in the dataset")
        if label_idxs_attr in self.__dict__:
            raise ValueError(f"Label {label_name} already exists in the dataset")

        setattr(self, label_names_attr, names)
        setattr(self, label_name_to_idx_attr, {c: i for i, c in enumerate(names)})
        setattr(self, label_idxs_attr, indices)

        self._BUILTIN_LABELS.add(label_name)

    def __getitem__(self, index: int) -> dict:
        path = self.images[index]
        image_pil = self.loader(path) if isinstance(path, str) else self.pairs_loader(*path)

        data = dict(
            _parent=self,
            images_fp=path,
            images_pil=image_pil,
            images_tensor=to_tensor(image_pil),
        )

        for label in sorted(self._BUILTIN_LABELS):
            label_idx = getattr(self, f"labels_{label}_idx")[index]
            label_name = getattr(self, f"{label}_names")[label_idx]
            label_tensor = torch.zeros(len(getattr(self, f"{label}_names")), dtype=torch.long)
            label_tensor[label_idx] = 1
            data[f"labels_{label}_idx"] = label_idx
            data[f"labels_{label}_name"] = label_name
            data[f"labels_{label}_tensor"] = label_tensor

        return data

    def __len__(self) -> int:
        return len(self.images)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")

        for label in sorted(self._BUILTIN_LABELS):
            label_names = getattr(self, f"{label}_names")
            if len(label_names) > 10:
                body.append(f"{label.capitalize()} names: {', '.join(label_names[:10])}...")
            else:
                body.append(f"{label.capitalize()} names: {', '.join(label_names)}")
        lines = [head] + ["    " + line for line in body]
        return "\n".join(lines)


class ExperimentDataset(ImageDataset):
    """Dataset of images for experiments.

    Args:
    ----
        root (PathLike): Root directory of dataset where `images` are found.
        kwargs: Extra arguments to pass to the dataset or dataloader.

    """

    def __init__(self, root: PathLike, **kwargs) -> None:
        images = _default_dict_file_loader(Path(root, "images.txt"))
        images["target_fp"] = [str(Path(root, v)) for v in images["target_fp"]]

        super().__init__(root=root, images=images["target_fp"], **kwargs)

        if "source_fp" in images:
            images["source_fp"] = [str(Path(root, v)) for v in images["source_fp"]]
            images = list(zip(images["source_fp"], images["target_fp"], strict=True))
            self.images = images

        # register labels
        labels_names_fp = list(Path(root).rglob("*_names.txt"))
        labels = [file.stem.split("_")[0] for file in labels_names_fp]
        labels_idxs = _default_dict_file_loader(Path(root, "labels_idxs.txt"))
        for label, label_names_fp in zip(labels, labels_names_fp, strict=True):
            names = _default_list_file_loader(label_names_fp)
            indices = [int(v) for v in labels_idxs[label]]
            self.register_labels(label, names, indices)

    @staticmethod
    def from_dict(
        root: PathLike,
        source: ImageDataset,
        samples_per_class: int = 10,
        **experiment_cfg: DictConfig | dict,
    ) -> "ExperimentDataset":
        """Generate an experiment dataset from a configuration.

        Args:
        ----
            root (PathLike): The root directory where the dataset will be generated.
            source (ImageDataset): The dataset to use for the experiment dataset.
            samples_per_class (int): The number of samples per class to generate
            experiment_cfg (DictConfig | dict): The configuration for the experiment dataset.

        """
        root = Path(root)

        # check if the experiment datasets already exist
        avail_configs = list(root.parent.glob("*/config.yaml"))
        ignore_fields = ["question"]
        experiment_cfg_without_ignored = {
            k: v for k, v in experiment_cfg.items() if k not in ignore_fields
        }
        exists_at = None
        for config_fp in avail_configs:
            config = OmegaConf.load(config_fp)
            config_without_ignored = {k: v for k, v in config.items() if k not in ignore_fields}
            if config_without_ignored == experiment_cfg_without_ignored:
                exists_at = config_fp.parent
                break

        if exists_at is None:
            ExperimentDataset.generate(root, source, samples_per_class, **experiment_cfg)
            OmegaConf.save(experiment_cfg, root / "config.yaml")
        else:
            root = exists_at

        return ExperimentDataset(root)

    @staticmethod
    def generate(
        root: PathLike,
        source: ImageDataset,
        samples_per_class: int = 10,
        **experiment_cfg: DictConfig | dict,
    ) -> None:
        """Generate a case study dataset from a configuration.

        Args:
        ----
            root (PathLike): The output directory where the dataset will be generated.
            source (ImageDataset): The dataset to use for the task dataset.
            samples_per_class (int): The number of samples per class to generate. Defaults to 10.
            experiment_cfg (DictConfig | dict): The configuration for the case dataset.

        """
        image_type = experiment_cfg.get("image_type")
        answers_cfg = experiment_cfg.get("vqa_answers")
        answers = [answer["text"].lower() for answer in answers_cfg]
        max_samples = samples_per_class * len(answers)
        select_function_cfg = [[answer["image_select_function"]] for answer in answers_cfg]
        transform_function_cfg = [[answer["image_transform_function"]] for answer in answers_cfg]

        # get the original images and the registered labels
        labels = list(source._BUILTIN_LABELS)

        # generate the dataset
        images_info_body, labels_info_body = [], []
        labels = list(sorted(["answer"] + labels))
        select_functions = ExperimentDataset._make_functions_from_cfg(select_function_cfg)
        transform_functions = ExperimentDataset._make_functions_from_cfg(transform_function_cfg)
        iterable = utils.get_iterable(source, desc="Generating data", total=max_samples)
        for i in range(max_samples):
            # sample an answer
            sample_answer = np.random.choice(answers)
            sample_answer_idx = answers.index(sample_answer)
            select_fn = select_functions[sample_answer_idx]
            transform_fn = transform_functions[sample_answer_idx]

            # select and transform the sample
            source_sample = select_fn(source)
            target_sample = transform_fn(source_sample)

            # update the labels
            answers_tensor = torch.zeros(len(answers), dtype=torch.long)
            answers_tensor[sample_answer_idx] = 1
            target_sample["labels_answer_idx"] = sample_answer_idx
            target_sample["labels_answer_name"] = sample_answer
            target_sample["labels_answer_tensor"] = answers_tensor

            # save sample to disk
            folder_name, filename = target_sample["images_fp"].split("/")[-2:]
            folder_dir = root / "target" / folder_name
            folder_dir.mkdir(parents=True, exist_ok=True)
            target_image_fp = folder_dir / filename
            target_sample["images_pil"].save(target_image_fp)

            # add the image info to the list
            if experiment_cfg.get("image_type") in ("single"):
                target_image_fp_stem = Path("target") / folder_name / filename
                images_info_body.append(f"{i},{target_image_fp_stem}")
            elif experiment_cfg.get("image_type") in ("pair"):
                source_image_fp_stem = Path("source") / folder_name / filename
                target_image_fp_stem = Path("target") / folder_name / filename
                images_info_body.append(f"{i},{source_image_fp_stem},{target_image_fp_stem}")

                # save the source image
                source_image_fp = root / source_image_fp_stem
                source_image_fp.parent.mkdir(parents=True, exist_ok=True)
                source_image_fp.symlink_to(Path(source_sample["images_fp"]).resolve())
            else:
                raise NotImplementedError("Unknown task dataset type")

            # add the labels info to the list
            label_idxs = [str(target_sample.get(f"labels_{label}_idx", -1)) for label in labels]
            labels_info_body.append(",".join([str(i), *label_idxs]))

            iterable.update(1)
        iterable.close()

        # save the images
        with open(root / "images.txt", "w") as f:
            header = "idx,target_fp\n" if image_type in ("single") else "idx,source_fp,target_fp\n"
            f.write(header)
            f.writelines("\n".join(images_info_body))

        # save the images labels idxs
        with open(root / "labels_idxs.txt", "w") as f:
            header = f"idx{','.join([''] + labels)}\n"
            f.write(header)
            f.writelines("\n".join(labels_info_body))

        # save the names of the labels
        for label in labels:
            label_names = answers if label == "answer" else getattr(source, f"{label}_names")
            with open(root / f"{label}_names.txt", "w") as f:
                f.writelines("\n".join(label_names))

    @staticmethod
    def _make_functions_from_cfg(tools_cfg: Sequence[Sequence[str | Any]]) -> list[Callable]:
        """Instantiate a list of function tools from config.

        Args:
        ----
            tools_cfg (Sequence[Sequence[str | Any]]): The list of tools to use for the task
                dataset. The keys are the answer names and the values are the tools to use for the
                task dataset.

        """
        tools = []

        # instantiate the transforms
        for tool_cfg in tools_cfg:
            function_instances = []
            for function in tool_cfg:
                function_call = function["full_name"]
                function_kwargs = function["kwargs"] or {}
                function_cfg = DictConfig({"_target_": function_call, **function_kwargs})
                function_instances.append(hydra.utils.instantiate(function_cfg))
            tools.append(Compose(function_instances))

        return tools


def _default_list_file_loader(fp: PathLike) -> list[str]:
    """Load a file from a path.

    Args:
    ----
        fp (PathLike): File path.

    """
    with open(fp) as f:
        data = f.read().split("\n")
        out = [line for line in data if line]
    return out


def _default_dict_file_loader(fp: PathLike) -> dict[str, str]:
    """Load a file from a path.

    Args:
    ----
        fp (PathLike): File path.

    """
    with open(fp) as f:
        data = f.read().split("\n")
        header, body = data[0], data[1:]
        header = header.split(",")
        body = [line.split(",") for line in body]
        body = list(zip(*body, strict=True))  # transpose the body

    return dict(zip(header, body, strict=True))


def _default_image_loader(path: PathLike) -> PILImage:
    """Load an image from a path.

    Args:
    ----
        path (PathLike): Path to the image.

    """
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def _default_paired_image_loader(
    source_image_fp: PathLike,
    target_image_fp: PathLike,
    axis: int = 0,
    background_color: tuple[int, int, int] = (255, 255, 255),
    resize: bool = True,
    margin: Literal["none", "square"] = "square",
    padding: int = 20,
) -> PILImage:
    """Load two images into a single image.

    Args:
    ----
        source_image_fp (PathLike): Path to the first image.
        target_image_fp (PathLike): Path to the second image.
        axis (int): The axis of concatenation. Default to 0.
        background_color (tuple[int, int, int]): The background color of the generated super
            image. Default to (255, 255, 255).
        resize (bool): Whether to rescale the images to have similar dimensions. Default to True.
        square_aspect_ratio (bool): Guarantee a squared aspect ratio.
        margin ("none" | "square"): The type of margin to apply to the created image. The "square"
            value guarantee a squared aspect ratio. This is useful in situations where a vision
            encoder performs center crop (a common scenario). Default to "square".
        padding (int): Padding dividing the two images. Default to 20.

    """
    source_image_pil = Image.open(source_image_fp)
    target_image_pil = Image.open(target_image_fp)

    if axis == 0:  # vertical
        width = max(source_image_pil.width, target_image_pil.width)
        height = source_image_pil.height + target_image_pil.height + padding
    elif axis == 1:  # horizontal
        width = source_image_pil.width + target_image_pil.width + padding
        height = max(source_image_pil.height, target_image_pil.height)
    else:
        raise ValueError("axis must be 0 or 1")

    if resize and axis == 0:
        source_image_pil = source_image_pil.resize(
            (width, int(source_image_pil.height * (width / source_image_pil.width)))
        )
        target_image_pil = target_image_pil.resize(
            (width, int(target_image_pil.height * (width / target_image_pil.width)))
        )
        height = source_image_pil.height + target_image_pil.height + padding
    elif resize and axis == 1:
        source_image_pil = source_image_pil.resize(
            (int(source_image_pil.width * (height / source_image_pil.height)), height)
        )
        target_image_pil = target_image_pil.resize(
            (int(target_image_pil.width * (height / target_image_pil.height)), height)
        )
        width = source_image_pil.width + target_image_pil.width + padding

    if margin == "none":
        pass
    elif margin == "square":
        size = max(width, height)
        width, height = size, size
    else:
        raise ValueError("margin must be 'none' or 'square'")

    image_cat = Image.new("RGB", (width, height))
    image_cat.paste(background_color, box=(0, 0, width, height))

    if axis == 0:
        source_box = ((width - source_image_pil.width) // 2, 0)
        target_box = ((width - target_image_pil.width) // 2, source_image_pil.height + padding)
        image_cat.paste(source_image_pil, source_box)
        image_cat.paste(target_image_pil, target_box)
    elif axis == 1:
        source_box = (0, (height - source_image_pil.height) // 2)
        target_box = (source_image_pil.width + padding, (height - target_image_pil.height) // 2)
        image_cat.paste(source_image_pil, source_box)
        image_cat.paste(target_image_pil, target_box)

    return image_cat
