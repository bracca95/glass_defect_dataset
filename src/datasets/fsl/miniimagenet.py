import os

from glob import glob
from typing import List, Set, Tuple, Optional
from torch.utils.data import Dataset

from .dataset_fsl import FewShotDataset
from ...utils.tools import Logger, Tools
from ...utils.config_parser import DatasetConfig
from ....config.consts import General as _CG


class MiniImagenet(FewShotDataset):
    """MiniImagenet

    This class takes for granted that:
        * each class is represented by a directory (100 classes)
        * each directory contains 600 samples
        * all the (100) class directories are located under the same root 

    SeeAlso:
        [main page](https://github.com/fiveai/on-episodes-fsl)
        [splits](https://github.com/mileyan/simple_shot/tree/master/split/mini)
        [download](https://drive.google.com/open?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE)
    """

    N_IMG_PER_CLASS = 600
    N_CLASSES_TRAIN = 64
    N_CLASSES_VAL = 16
    N_CLASSES_TEST = 20

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)

    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        return glob(os.path.join(self.dataset_config.dataset_path, "n*", "*JPEG"))
    
    def expected_length(self) -> int:
        return (self.N_CLASSES_TRAIN + self.N_CLASSES_TEST + self.N_CLASSES_VAL) * self.N_IMG_PER_CLASS
    
    def split_method(self) -> Tuple[Set[str], Set[str], Set[str]]:
        import pandas as pd

        path = os.path.join(os.path.dirname(self.dataset_config.dataset_path), "preprocessing", "miniimagenet", "ultimate_split")
        path = Tools.validate_path(path)

        def get_class_set(split_name: str):
            split_path = Tools.validate_path(os.path.join(path, f"{split_name}.csv"))
            df = pd.read_csv(split_path)
            return set(df["label"].values)
        
        return get_class_set("train"), get_class_set("val"), get_class_set("test")
    
    @property
    def image_list(self) -> List[str]:
        return self._image_list
    
    @image_list.setter
    def image_list(self, value):
        self._image_list = value

    @property
    def label_list(self) -> List[int]:
        return self._label_list
    
    @label_list.setter
    def label_list(self, value):
        self._label_list = value

    @property
    def train_dataset(self) -> Dataset:
        return self._train_dataset
    
    @train_dataset.setter
    def train_dataset(self, value: Dataset):
        self._train_dataset = value

    @property
    def test_dataset(self) -> Dataset:
        return self._test_dataset
    
    @test_dataset.setter
    def test_dataset(self, value: Dataset):
        self._test_dataset = value

    @property
    def val_dataset(self) -> Optional[Dataset]:
        return self._val_dataset
    
    @val_dataset.setter
    def val_dataset(self, value: Optional[Dataset]):
        self._val_dataset = value