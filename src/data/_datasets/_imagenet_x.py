from collections.abc import Sequence
from pathlib import Path

from src import utils
from src.data._datasets._api import register_dataset
from src.data._datasets._imagenet import ImageNet
from src.utils.types import DataFrame, PathLike

log = utils.get_logger(__name__, rank_zero_only=True)

__all__ = ["ImageNetX"]


@register_dataset("imagenet-x")
class ImageNetX(ImageNet):
    """ImageNet-X dataset.

    Args:
    ----
        root (PathLike): Root directory of dataset where `images` are found.
        split (str, optional): The split of the dataset to use. Defaults to "train".
        download (bool, optional): Whether to download the dataset. Defaults to False.

    Extra args:
        which_factor (str, optional): Which factors to use for the annotations, either "top" or
            "multi". Default to "top".

    Attributes:
    ----------
        attribute_names (Sequence): List of the attribute names.
        attribute_to_idx (dict): Dict with items (attribute_name, attribute_index).
        class_names (Sequence): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        group_names (Sequence): List of the group names.
        group_to_idx (dict): Dict with items (group_name, group_index).
        images (Sequence): List of paths to images.
        labels_attribute_idx (Sequence): List of attributes for each image in the dataset.
        labels_class_idx (Sequence): The class index value for each image in the dataset.
        labels_group_idx (Sequence): The group index value for each image in the dataset.

    """

    _AVAILABLE_SPLITS: Sequence[str] = ["train", "val", "prototypes"]
    _AVAILABLE_IMAGE_TYPES: Sequence[str] = ["photo"]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert utils.IMAGENET_X_AVAILABLE, "imagenet-x package not installed"
        from imagenet_x import FACTORS, METACLASSES

        annotations = self.load_annotations(
            split=self.split, which_factor=kwargs.pop("which_factor", "top")
        )
        filenames = [path.split("/")[-1] for path in self.images]
        mask = [filename in annotations.index for filename in filenames]

        self.filter(mask=mask, in_place=True)

        filenames = [path.split("/")[-1] for path in self.images]
        attribute_names = [name.replace("_", " ") for name in FACTORS]
        labels_attribute_idx = annotations.loc[filenames][FACTORS].values.astype(bool)
        labels_attribute_idx = [
            i for j in range(len(self)) for i, a in enumerate(labels_attribute_idx[j]) if a == 1
        ]
        self.register_labels("attribute", attribute_names, labels_attribute_idx)

        labels_group_names = annotations.loc[filenames]["metaclass"].values.tolist()
        labels_group_names = [name.replace("_", " ") for name in labels_group_names]
        group_names = [name.replace("_", " ") for name in METACLASSES]
        group_to_idx = {name: idx for idx, name in enumerate(group_names)}
        labels_group_idx = [group_to_idx[name] for name in labels_group_names]
        self.register_labels("group", group_names, labels_group_idx)

    @staticmethod
    def load_annotations(split: str, which_factor: str = "top") -> DataFrame:
        """Load the ImageNet-X annotations.

        Args:
        ----
            split (str): The split of the dataset to use.
            which_factor (str): Which factors to use for the annotations, either "top" or "multi".
                Default to "top".

        """
        assert utils.IMAGENET_X_AVAILABLE, "imagenet-x package not installed"
        from imagenet_x import load_annotations

        if split in ["train", "val"]:
            annotations = load_annotations(
                which_factor=which_factor,
                partition=split,
                filter_prototypes=True,
            ).set_index("file_name")
        elif split == "prototypes":
            annotations_without_prototypes = load_annotations(
                which_factor=which_factor,
                partition="val",
                filter_prototypes=True,
            ).set_index("file_name")
            annotations_with_prototypes = load_annotations(
                which_factor=which_factor,
                partition="val",
                filter_prototypes=False,
            ).set_index("file_name")

            annotations = annotations_with_prototypes[
                ~annotations_with_prototypes.index.isin(annotations_without_prototypes.index)
            ]

        return annotations

    @staticmethod
    def download(root: PathLike = "data/imagenet-x", exist_ok: bool = True) -> None:
        """Download the ImageNet-X.

        Args:
        ----
            root (PathLike): The output directory where the dataset will be downloaded.
                Defaults to "data/imagenet-x".
            exist_ok (bool): Whether to raise an error if the output directory
                already exists. Defaults to True.

        """
        dataset_dir = Path(root)
        data_dir = dataset_dir.parent

        ImageNet.download(exist_ok=True)
        imagenet_dir = Path(data_dir, "imagenet")

        if dataset_dir.exists() and not exist_ok:
            raise FileExistsError(f"{dataset_dir} already exists")
        elif dataset_dir.exists() and exist_ok:
            return

        # create the symbolic links from the imagenet dataset
        for split in ["train", "val", "prototypes"]:
            annotations = ImageNetX.load_annotations(split=split)

            imagenet_split = "train" if split == "train" else "val"
            imagenet = ImageNet(root=imagenet_dir, split=imagenet_split)

            # create the class folders
            folder_names = [Path(image).parent.name for image in imagenet.images]
            folder_names = sorted(set(folder_names))
            for folder_name in folder_names:
                folder_dir = Path(dataset_dir, split, folder_name)
                folder_dir.mkdir(parents=True, exist_ok=True)

            # filter the images
            filenames = [path.split("/")[-1] for path, _ in imagenet.samples]
            mask = [filename in annotations.index for filename in filenames]
            images = [image for image, keep in zip(imagenet.images, mask, strict=True) if keep]

            # create the symbolic links
            target_dir = Path(dataset_dir, split)
            target_dir.mkdir(parents=True, exist_ok=True)
            for image in images:
                source = Path(image).resolve()
                folder_name, filename = image.split("/")[-2:]
                target = Path(target_dir, folder_name, filename)
                target.symlink_to(source)
