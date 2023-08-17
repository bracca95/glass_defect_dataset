# about TypedDict https://stackoverflow.com/a/64938100
import torch
from torch.utils.data import Subset
from typing import Optional, TypeVar, TypedDict

T = TypeVar("T")
SubsetsDict = TypedDict("SubsetsDict", { "train": Subset, "val": Optional[Subset], "test": Subset })


class General:
    DEFAULT_BOOL = False
    DEFAULT_INT = 0
    DEFAULT_STR = ""
    DEFAULT_LIST = []
    DEFAULT_SUBSETS = ["train", "val", "test"]
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    EPS = 0.001


class DatasetConst:
    CONFIG_DATASET_PATH = "dataset_path"
    CONFIG_DATASET_TYPE = "dataset_type"
    CONFIG_DATASET_SPLITS = "dataset_splits"
    CONFIG_AUGMENT_ONLINE = "augment_online"
    CONFIG_AUGMENT_OFFLINE = "augment_offline"
    CONFIG_CROP_SIZE = "crop_size"
    CONFIG_IMAGE_SIZE = "image_size"
    CONFIG_DATASET_MEAN = "dataset_mean"
    CONFIG_DATASET_STD = "dataset_std"