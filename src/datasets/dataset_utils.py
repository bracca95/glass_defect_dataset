from typing import Union
from torch.utils.data import Dataset

from .dataset import DatasetWrapper
from .other.glass_plate import GlassPlate, GlassPlateTrainYolo, GlassPlateTestYolo
from .other.glass_plate_inference import DefectOptDataset
from .custom.defectviews import GlassOpt, GlassOptBckg, GlassOptTricky, GlassOptDouble, GlassOptDoubleInference, GlassOptFullInference, BubblePoint, QPlusV1, QPlusV2, QPlusDouble
from .fsl.omniglot import OmniglotWrapper
from .fsl.miniimagenet import MiniImagenet
from .fsl.cifar import CifarFs
from .fsl.cub import Cub
from .torch.celeba import CelebaWrapper
from ..utils.config_parser import DatasetConfig
from ..utils.tools import Logger


err_msg = "values allowed: {`opt6`, `opt_bckg`, `opt_double`, `opt_double_inference`, `binary`, `qplusv1`, " + \
    "`qplusv2`, `qplus_double`, `omniglot`, `miniimagenet`, `opt_yolo_train`, `opt_yolo_test`, `cub`, " + \
    "`cifar_fs`, `celeba`, `full_inference`} for dataset_type"


class DatasetBuilder:

    @staticmethod
    def load_dataset(dataset_config: DatasetConfig) -> Union[DatasetWrapper, Dataset]:
        if dataset_config.dataset_type == "opt6":
            Logger.instance().info("Loading dataset GlassOpt (type DatasetWrapper)")
            return GlassOpt(dataset_config)
        elif dataset_config.dataset_type == "opt_bckg":
            Logger.instance().info("Loading dataset GlassOptBckg (type GlassOpt)")
            return GlassOptBckg(dataset_config)
        elif dataset_config.dataset_type == "opt_tricky":
            Logger.instance().info("Loading dataset GlassOptTricky (type GlassOpt)")
            return GlassOptTricky(dataset_config)
        elif dataset_config.dataset_type == "opt_double":
            Logger.instance().info("Loading dataset GlassOptDouble (type GlassOpt)")
            return GlassOptDouble(dataset_config)
        elif dataset_config.dataset_type == "opt_double_inference":
            Logger.instance().info("Loading dataset GlassOptDoubleInference (type GlassOptDouble)")
            return GlassOptDoubleInference(dataset_config)
        elif dataset_config.dataset_type == "full_inference":
            Logger.instance().info("Loading dataset GlassOptFullInference (type GlassOptDoubleInference)")
            return GlassOptFullInference(dataset_config)
        elif dataset_config.dataset_type == "qplus_double":
            Logger.instance().info("Loading dataset QPlusDouble (type GlassOptDouble)")
            return QPlusDouble(dataset_config)
        elif dataset_config.dataset_type == "qplusv1":
            Logger.instance().info("Loading dataset QPlusV1 (type GlassOpt)")
            return QPlusV1(dataset_config)
        elif dataset_config.dataset_type == "qplusv2":
            Logger.instance().info("Loading dataset QPlusV2 (type GlassOpt)")
            return QPlusV2(dataset_config)
        elif dataset_config.dataset_type == "binary":
            Logger.instance().info("Loading dataset BubblePoint (type GlassOpt)")
            return BubblePoint(dataset_config)
        elif dataset_config.dataset_type == "omniglot":
            Logger.instance().info("Loading dataset Omniglot (type FewShotDataset)")
            return OmniglotWrapper(dataset_config)
        elif dataset_config.dataset_type == "miniimagenet":
            Logger.instance().info("Loading dataset Mini Imagenet (type FewShotDataset)")
            return MiniImagenet(dataset_config)
        elif dataset_config.dataset_type == "cifar_fs":
            Logger.instance().info("Loading dataset CIFAR-FS (type FewShotDataset)")
            return CifarFs(dataset_config)
        elif dataset_config.dataset_type == "cub":
            Logger.instance().info("Loading dataset CUB (type FewShotDataset)")
            return Cub(dataset_config)
        elif dataset_config.dataset_type == "celeba":
            Logger.instance().info("Loading dataset Omniglot (type Dataset)")
            return CelebaWrapper(dataset_config)
        else:
            raise ValueError(err_msg)
        
class YoloDatasetBuilder:

    @staticmethod
    def load_dataset(dataset_config: DatasetConfig) -> GlassPlate:
        if dataset_config.dataset_type == "opt_yolo_train":
            Logger.instance().info("Loading dataset GlassPlateTrainYolo (type GlassPlate)")
            return GlassPlateTrainYolo(dataset_config)
        elif dataset_config.dataset_type == "opt_yolo_test":
            Logger.instance().info("Loading dataset GlassPlateTestYolo (type GlassPlate)")
            return GlassPlateTestYolo(dataset_config)
        else:
            raise ValueError(err_msg)


class FullInferenceBuilder:

    @staticmethod
    def load_dataset(dataset_config: DatasetConfig) -> DefectOptDataset:
        if dataset_config.dataset_type == "full_inference":
            Logger.instance().info("Loading dataset FullInference")
            return DefectOptDataset(dataset_config)
        
        raise ValueError(err_msg)
