import os
import torch

from PIL import Image
from glob import glob
from typing import Callable, List, Optional
from torchvision import transforms

from .custom_dataset import CustomDataset
from ..dataset import DatasetLauncher, InferenceLauncher
from ...imgproc import Processing
from ...utils.tools import Logger, Tools
from ...utils.config_parser import DatasetConfig


class GlassOpt(CustomDataset):

    label_to_idx = {
        "bubble": 0, 
        "point": 1,
        "break": 2,
        "dirt": 3,
        "mark": 4,
        "scratch": 5
    }

    idx_to_label = Tools.invert_dict(label_to_idx)

    AUG_DIR = "img_augment"
    NO_CROP = ["scratch", "break", "mark"]

    # https://stackoverflow.com/a/42583719
    # -1 means split all the occurrences of the split character(s). n means split n words starting from the last
    split_name = staticmethod(lambda x: os.path.basename(x).rsplit("_", -1)[0])

    @property
    def augment_strategy(self):
        return self._augment_strategy
    
    @augment_strategy.setter
    def augment_strategy(self, val):
        self._augment_strategy = val

    def __init__(self, dataset_config: DatasetConfig):
        self.augment_strategy = Processing.offline_transforms_v2
        super().__init__(dataset_config)

    def get_image_list(self, filt: List[str]) -> List[str]:
        """Read all the filenames in the dataset directory

        Read and return all the filenames in the dataset directory. If filt is not None, it will read only the specified
        elements (by class names - `self.label_to_idx` as an example). It will read also the directory containing
        augmented images, if initialized.

        Args:
            filt (Optional[List[str]]): if not None, restrict the defect classes by name

        Returns:
            List[str] containing the full path to the images
        """
        
        extra_image_list = glob(os.path.join(self.dataset_aug_path, "*.png")) if self.dataset_config.augment_offline else list()
        image_list = glob(os.path.join(self.dataset_config.dataset_path, "*.png"))
        image_list = image_list + extra_image_list
        
        image_list = list(filter(lambda x: Tools.check_string(self.split_name(x), filt, True, True), image_list))
        
        if not all(map(lambda x: x.endswith(".png"), image_list)) or image_list == []:
            raise ValueError("incorrect image list. Check the provided path for your dataset.")
        
        Logger.instance().info("Got image list")

        return image_list

    def get_label_list(self) -> List[int]:
        """Get all the label names, for each image name

        Returns:
            List[str] with the name of the labels
        """
        
        if len(self.image_list) == 0:
            self.get_image_list(self.filt)

        label_list = list(map(lambda x: self.split_name(x), self.image_list))
       
        Logger.instance().debug(f"Labels used: {set(label_list)}")
        Logger.instance().debug(f"Number of images per class: { {i: label_list.count(i) for i in set(label_list)} }")

        return [self.label_to_idx[defect] for defect in label_list]
        

    def load_image(self, path: str, augment: bool) -> torch.Tensor:
        """Load image as tensor

        This will be used in the trainloader only. Read the image, crop it, resize it (if required in config), make it
        a tensor and normalize (if required).

        Args:
            path (str): full path to the image

        Returns:
            torch.Tensor of the modified image

        See Also:
            https://stackoverflow.com/questions/51677788/data-augmentation-in-pytorch
            https://stackoverflow.com/questions/51677788/data-augmentation-in-pytorch
            https://stackoverflow.com/a/72642001
        """
        
        img_pil = Image.open(path).convert("L")

        if augment:
            # TODO implement
            pass

        # crop
        # augmented images are already square-shaped, do not crop!
        # scratches, breaks and marks are likely not to be square-shaped, so cropping will lose information
        # check: I could crop only bubbles, since they are the only ones that have to preserve proportions strictly
        if self.dataset_aug_path not in path and not Tools.check_string(os.path.basename(path), self.NO_CROP, False, False):
            img_pil = Processing.crop_no_padding(img_pil, self.dataset_config.crop_size, path)
        
        # resize
        img_pil = transforms.Resize((self.dataset_config.image_size, self.dataset_config.image_size))(img_pil)

        # rescale [0-255](int) to [0-1](float)
        img = transforms.ToTensor()(img_pil)

        # normalize
        img = DatasetLauncher.normalize_or_identity(self.dataset_config)(img)

        return img # type: ignore


class GlassOptBckg(GlassOpt):

    label_to_idx = {
        "background": 0,
        "bubble": 1, 
        "point": 2,
        "break": 3,
        "dirt": 4,
        "mark": 5,
        "scratch": 6
    }

    idx_to_label = Tools.invert_dict(label_to_idx)
    NO_CROP = ["scratch", "break", "mark", "background"]

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)


class QPlusV1(GlassOpt):

    label_to_idx = {
        "mark": 0,
        "glass_id": 1,
        "dirt": 2,
        "point": 3,
        "scratch_light": 4,
        "halo": 5,
        "scratch_multi": 6,
        "dust": 7,
        "coating": 8,
        "scratch_heavy": 9,
        "dirt_halo": 10
    }

    idx_to_label = Tools.invert_dict(label_to_idx)
    NO_CROP = ["mark", "glass_id", "dirt", "dirt_halo", "point", "scratch_light", "scratch_heavy", "scratch_multi", "dust", "halo", "coating"]
    split_name = staticmethod(lambda x: os.path.basename(x).rsplit("_did", 1)[0])

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)


class QPlusV2(GlassOpt):

    label_to_idx = {
        "mark": 0,
        "glass_id": 1,
        "dirt": 2,
        "point": 3,
        "scratch_light": 4,
        "halo": 5,
        "scratch_multi": 6,
        "dust": 7,
        "coating": 8,
        "scratch_heavy": 9,
        "bubble": 10,
        "bubble_hole": 11
    }

    # "dirt_halo": 12,

    idx_to_label = Tools.invert_dict(label_to_idx)
    NO_CROP = list(label_to_idx.keys())
    split_name = staticmethod(lambda x: os.path.basename(x).rsplit("_did", 1)[0])

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)


class BubblePoint(GlassOpt):

    label_to_idx = {
        "bubble": 0,
        "point": 1
    }

    idx_to_label = Tools.invert_dict(label_to_idx)

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)


class GlassOptTricky(GlassOpt):

    label_to_idx = {
        "background": 0,
        "bubble": 1, 
        "point": 2,
        "dirt": 3
    }

    idx_to_label = Tools.invert_dict(label_to_idx)

    NO_CROP = list(label_to_idx.keys())     # ["background", "scratch", "dirt"]
    split_name = staticmethod(lambda x: os.path.basename(x).rsplit("_", -1)[0])

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)


class GlassOptDouble(GlassOpt):

    """Cropping here makes no sense if the defect is not centered!
    {
        'break': 2,
        'bubble': 1740,
        'dirt': 308,
        'dirt_point_small': 36,
        'mark': 6,
        'point': 282,
        'scratch_heavy': 160
    }
    """

    label_to_idx = {
        "bubble": 0,
        "point": 1,
        "dirt": 2,
        "scratch_heavy": 3
    }

    idx_to_label = Tools.invert_dict(label_to_idx)

    NO_CROP = ["scratch_heavy"] # list(label_to_idx.keys())  # ["scratch_heavy", "dirt"]
    split_name = staticmethod(lambda x: os.path.basename(x).rsplit("_did", 1)[0])

    # use only one channel (only for test purpose)
    TEST_ONE = False

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)

    # TODO: offline aumentation. For the moment, avoid augmentation as in Tricky (augs are random, must apply the same to both channels)
    def get_image_list(self, filt: List[str]) -> List[str]:
        image_list = glob(os.path.join(self.dataset_config.dataset_path, "*.png"))
        image_list = list(filter(lambda x: x.endswith("vid_1.png"), image_list))
        image_list = list(filter(lambda x: Tools.check_string(self.split_name(x), filt, True, True), image_list))
        
        if not all(map(lambda x: x.endswith(".png"), image_list)) or image_list == []:
            raise ValueError("incorrect image list. Check the provided path for your dataset.")
        
        Logger.instance().info("Got image list")

        return image_list
    
    def load_image(self, path: str, augment: bool) -> torch.Tensor:
        path_2 = path.replace("_vid_1.png", "_vid_2.png")

        img_pil_1 = Image.open(path).convert("L")
        img_pil_2 = Image.open(path_2).convert("L")

        # # DEBUG - may need to save test images for further tests
        # root = "/media/lorenzo/M/datasets/dataset_opt/2.2_dataset_opt/dmx_2c/test_samples_1"
        # img_pil_1.save(os.path.join(root, os.path.basename(path)))
        # img_pil_2.save(os.path.join(root, os.path.basename(path_2)))

        if augment:
            # TODO implement
            pass

        # crop
        if not Tools.check_string(os.path.basename(path), self.NO_CROP, False, False):
            img_pil_1 = Processing.crop_no_padding(img_pil_1, self.dataset_config.crop_size, path)
            img_pil_2 = Processing.crop_no_padding(img_pil_2, self.dataset_config.crop_size, path)
        
        # resize
        img_pil_1 = transforms.Resize((self.dataset_config.image_size, self.dataset_config.image_size))(img_pil_1)
        img_pil_2 = transforms.Resize((self.dataset_config.image_size, self.dataset_config.image_size))(img_pil_2)

        # rescale [0-255](int) to [0-1](float)
        img_1 = transforms.ToTensor()(img_pil_1)
        img_2 = transforms.ToTensor()(img_pil_2)

        img = torch.stack([img_1, img_2], dim=0).squeeze(1)

        # if use only one channel for test
        if self.TEST_ONE:
            del img
            img = img_1.clone()

        # normalize
        img = DatasetLauncher.normalize_or_identity(self.dataset_config)(img)

        return img # type: ignore
    

class GlassOptDoubleInference(GlassOptDouble):
    """OPT with two channels. Use only for precise inference"""

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)
        self.test_dataset = InferenceLauncher(self.test_dataset.image_list, self.test_dataset.label_list, False, self.load_image)


class GlassOptFullInference(GlassOptDoubleInference):

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)

    def get_image_list(self, filt: Optional[List[str]]=None) -> List[str]:
        image_list = glob(os.path.join(self.dataset_config.dataset_path, "*.png"))
        image_list = list(filter(lambda x: x.endswith("vid_1.png"), image_list))
        
        if not all(map(lambda x: x.endswith(".png"), image_list)) or image_list == []:
            raise ValueError("incorrect image list. Check the provided path for your dataset.")
        
        Logger.instance().info("Got image list")

        return image_list
    
    def get_label_list(self) -> List[int]:
        """Get all the label names, for each image name

        Returns:
            List[str] with the name of the labels
        """
        
        if len(self.image_list) == 0:
            self.get_image_list(self.filt)

        label_list = list(map(lambda x: self.split_name(x), self.image_list))
       
        Logger.instance().debug(f"Labels used: {set(label_list)}")
        Logger.instance().debug(f"Number of images per class: { {i: label_list.count(i) for i in set(label_list)} }")

        labels = set(label_list)
        if any(map(lambda x: x not in self.label_to_idx.keys(), labels)):
            self.label_to_idx = { k: v for v, k in enumerate(labels) }
            self.idx_to_label = Tools.invert_dict(self.label_to_idx)

        return [self.label_to_idx[defect] for defect in label_list]

class QPlusDouble(GlassOptDouble):

    """{
        'bubble': 196,
        'bubble_hole': 2,
        'bubble_small': 48,
        'dirt': 130,
        'dirt_halo': 6,
        'point': 276,
        'dirt_p_multi': 6,
        'dust': 18,
        'halo': 170,
        'halo_points': 100,
        'inclusion': 14,
        'mark': 2,
        'point_td': 2,
        'scratch_heavy': 2,
        'scratch_light': 138,
        'scratch_multi': 6
    }
    """

    label_to_idx = {
        "bubble": 0,
        "bubble_small": 1,
        "dirt": 2,
        "halo": 3,
        "halo_points": 4,
        "point": 5,
        "scratch_light": 6,
    }

    idx_to_label = Tools.invert_dict(label_to_idx)

    NO_CROP = ["scratch_heavy"] # list(label_to_idx.keys()) # ["scratch_heavy", "dirt"]
    split_name = staticmethod(lambda x: os.path.basename(x).rsplit("_did", 1)[0])

    # use only one channel (only for test purpose)
    TEST_ONE = False

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)