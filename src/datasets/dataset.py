import os
import torch

from abc import ABC, abstractproperty, abstractmethod
from typing import Tuple, List, Callable, Optional
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader

from ..utils.tools import Logger
from ..utils.config_parser import DatasetConfig


class DatasetWrapper(ABC):
    """Wrapper for an entire dataset

    This interface is meant to wrap an entire dataset with its splits. Torch Datasets are usually kept separated, but
    I want to wrap them in a unique class in order to be able to split as I wish and keep track of the label list and
    image paths for all the samples (it may be required in tasks like FSL).
    """

    @abstractproperty
    def image_list(self) -> List[str]:
        ...

    @abstractproperty
    def label_list(self) -> List[int]:
        ...

    @abstractproperty
    def train_dataset(self) -> Dataset:
        ...

    @abstractproperty
    def test_dataset(self) -> Dataset:
        ...

    @abstractproperty
    def val_dataset(self) -> Optional[Dataset]:
        ...


class DatasetLauncher(Dataset):
    """One split of DatasetWrapper

    This class is meant to replicate a single torch Dataset, but it is able to keep track of all the indices and image
    paths. Torch structures like ImageFolder or other subclasses of Dataset do not always allow for that.
    """

    def __init__(
            self,
            image_list: List[str],
            label_list: List[int],
            augment: bool,
            load_img_callback: Callable[[str, bool], torch.Tensor]
    ):
        super().__init__()
        self.image_list = image_list
        self.label_list = label_list
        self.augment = augment
        self.load_img_callback = load_img_callback

        self.info_dict: Optional[dict] = None

    def __getitem__(self, index):
        curr_img_batch = self.image_list[index]
        curr_label_batch = self.label_list[index]
        
        return self.load_img_callback(curr_img_batch, self.augment), curr_label_batch

    def __len__(self):
        return len(self.label_list)

    def set_info(self, label_list: List[int], idx_to_label: dict):
        t_label = torch.tensor(label_list, dtype=torch.int)
        info = { idx_to_label[value.item()]: (t_label == value).sum().item() for value in torch.unique(t_label) }
        self.info_dict = info

    @staticmethod
    def compute_mean_std(dataset: Dataset, ds_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if "imagenet" in ds_type:
            Logger.instance().debug(f"Dataset type is {ds_type}: imagenet mean/std selected")
            return torch.Tensor([0.485, 0.456, 0.406]), torch.Tensor([0.229, 0.224, 0.225])

        # https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/31
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        mean = 0.0
        for batch in dataloader:
            images, _ = batch
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
        mean = mean / len(dataloader.dataset)

        var = 0.0
        pixel_count = 0
        for batch in dataloader:
            images, _ = batch
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            var += ((images - mean.unsqueeze(1))**2).sum([0,2])
            pixel_count += images.nelement()
        std = torch.sqrt(var / pixel_count)

        if any(map(lambda x: torch.isnan(x), mean)) or any(map(lambda x: torch.isnan(x), std)):
            raise ValueError("mean or std are none")

        Logger.instance().warning(f"Mean: {mean}, std: {std}. Run the program again.")
        return mean, std
    
    @staticmethod
    def normalize_or_identity(dataset_config: DatasetConfig) -> torch.nn.Module:
        """Normalize image/s by dataset's mean and std

        If config allows normalization, first check if the mean and std values are available to perform the it. If not,
        returns the original image via identity function. This case can either be an error in configuration or it is 
        necessary when this method is called during the computation of the dataset's mean and std (load_image method).

        Args:
            dataset_config (..utils.config_parser.DatasetConfig)

        Returns:
            torch.nn.Module: Normalize module (with computed mean/std) or identity function
        """

        if dataset_config.normalize:
            if dataset_config.dataset_mean is not None and dataset_config.dataset_std is not None:
                normalize = transforms.Normalize(
                    torch.Tensor(dataset_config.dataset_mean),
                    torch.Tensor(dataset_config.dataset_std)
                )
                return normalize

        return torch.nn.Identity()
    
    @staticmethod
    def save_sample_image_batch(dataset: Dataset, outfolder: str):
        if "sample_batch.png" in os.listdir(outfolder):
            return
        
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        example_data, _ = next(iter(loader))
        img_grid = make_grid(example_data)

        grid_pil = transforms.ToPILImage()(img_grid)
        grid_pil.save(os.path.join(outfolder, "sample_batch.png"))


class InferenceLauncher(DatasetLauncher):
    """Only used in GlassOptDoubleInference

    This is a custom launcher to retrieve the image paths
    """

    def __init__(self, image_list: List[str], label_list: List[int], augment: bool, load_img_callback: Callable[[str, bool], torch.Tensor]):
        super().__init__(image_list, label_list, augment, load_img_callback)

    def __getitem__(self, index):
        curr_img_batch = self.image_list[index]
        curr_label_batch = self.label_list[index]
        
        return self.load_img_callback(curr_img_batch, False), curr_label_batch, curr_img_batch