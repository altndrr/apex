import zipfile
from collections.abc import Sequence
from pathlib import Path

from torchvision.models._meta import _IMAGENET_CATEGORIES
from tqdm import tqdm

from src import utils
from src.data._datasets._api import register_dataset
from src.data._datasets._base import ImageDataset
from src.utils.types import PathLike

log = utils.get_logger(__name__, rank_zero_only=True)

__all__ = ["ImageNet"]


@register_dataset("imagenet")
class ImageNet(ImageDataset):
    """ImageNet dataset.

    Args:
    ----
        root (PathLike): Root directory of dataset where `images` are found.
        split (str, optional): The split of the dataset to use. Defaults to "train".
        download (bool, optional): Whether to download the dataset. Defaults to False.

    Attributes:
    ----------
        class_names (Sequence): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        images (Sequence): List of paths to images.
        labels_class_idx (Sequence): The class index value for each image in the dataset.

    """

    _AVAILABLE_SPLITS: Sequence[str] = ["train", "val"]
    _AVAILABLE_IMAGE_TYPES: Sequence[str] = ["photo"]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        folder_names = {path.name for path in Path(self.root).glob("*")}
        folder_names = sorted(folder_names)
        folder_names_to_idx = {c: i for i, c in enumerate(folder_names)}
        class_names = _IMAGENET_CATEGORIES
        labels_class_idx = [folder_names_to_idx[Path(f).parent.name] for f in self.images]
        self.register_labels("class", class_names, labels_class_idx)

    @staticmethod
    def download(root: PathLike = "data/imagenet", exist_ok: bool = True) -> None:
        """Download the ImageNet dataset from Kaggle.

        Args:
        ----
            root (PathLike): The output directory where the dataset will be downloaded.
                Defaults to "data/imagenet".
            exist_ok (bool): Whether to raise an error if the output directory
                already exists. Defaults to True.

        """
        assert utils.KAGGLE_AVAILABLE, "kaggle package not installed"
        assert utils.WANDB_AVAILABLE, "pandas package not installed"

        import kaggle
        import pandas as pd

        dataset_dir = Path(root)
        data_dir = dataset_dir.parent

        if dataset_dir.exists() and not exist_ok:
            raise FileExistsError(f"{dataset_dir} already exists")
        elif dataset_dir.exists() and exist_ok:
            return

        try:
            kaggle.api.authenticate()
        except NameError as err:
            raise OSError("Could not find kaggle environmental variables") from err

        # download from kaggle
        kaggle.api.competition_download_files(
            "imagenet-object-localization-challenge",
            path=str(data_dir),
            quiet=False,
        )

        target = Path(data_dir, "imagenet-object-localization-challenge.zip")
        zip_ref = zipfile.ZipFile(target, "r")

        # extract and move train images to correct folder
        train_images = [f for f in zip_ref.namelist() if "ILSVRC/Data/CLS-LOC/train/" in f]
        for filename in tqdm(train_images, total=len(train_images)):
            zip_ref.extract(member=filename, path=Path(data_dir))
        output_folder = Path(data_dir, "ILSVRC/Data/CLS-LOC/train/")
        target_dir = Path(data_dir, "ImageNet", "train")
        target_dir.mkdir(parents=True, exist_ok=True)
        output_folder.rename(target_dir)

        # extract and read the val annotations
        zip_ref.extract(member="LOC_val_solution.csv", path=Path(data_dir, "ImageNet"))
        val_annotations = pd.read_csv(
            Path(data_dir, "ImageNet", "LOC_val_solution.csv"),
            header=0,
            index_col=0,
        )

        # # extract and move val images to correct folder
        val_images = [f for f in zip_ref.namelist() if "ILSVRC/Data/CLS-LOC/val/" in f]
        for filename in tqdm(val_images, total=len(val_images)):
            zip_ref.extract(member=filename, path=Path(data_dir))
            annotations = val_annotations.loc[[Path(filename).stem]]
            folder_name = annotations["PredictionString"].item().split(" ")[0]
            file = Path(data_dir, filename)
            folder_dir = Path(data_dir, "ImageNet", "val", folder_name)
            folder_dir.mkdir(parents=True, exist_ok=True)
            file.rename(Path(folder_dir, file.name))
        zip_ref.close()

        # remove the residue of the extraction
        Path(data_dir, "ILSVRC/Data/CLS-LOC/val").rmdir()
        Path(data_dir, "ILSVRC/Data/CLS-LOC").rmdir()
        Path(data_dir, "ILSVRC/Data").rmdir()
        Path(data_dir, "ILSVRC").rmdir()
        Path(data_dir, "ImageNet", "LOC_val_solution.csv").unlink()
