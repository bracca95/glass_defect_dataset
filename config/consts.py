# about TypedDict https://stackoverflow.com/a/64938100
import torch
from torch.utils.data import Subset
from typing import Optional, TypeVar, TypedDict

T = TypeVar("T")
PlatePathsDict = TypedDict("PlatePathsDict", { "ch_1": str, "ch_2": str })


class General:
    DEFAULT_BOOL = False
    DEFAULT_INT = 0
    DEFAULT_STR = ""
    DEFAULT_LIST = []
    DEFAULT_SUBSETS = ["train", "val", "test"]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPS = 0.001


class DatasetConst:
    CONFIG_DATASET_PATH = "dataset_path"
    CONFIG_DATASET_TYPE = "dataset_type"
    CONFIG_DATASET_SPLITS = "dataset_splits"
    CONFIG_NORMALIZE = "normalize"
    CONFIG_AUGMENT_ONLINE = "augment_online"
    CONFIG_AUGMENT_OFFLINE = "augment_offline"
    CONFIG_CROP_SIZE = "crop_size"
    CONFIG_IMAGE_SIZE = "image_size"
    CONFIG_DATASET_MEAN = "dataset_mean"
    CONFIG_DATASET_STD = "dataset_std"
    CONFIG_CSV_PATH = "csv_path"
    CONFIG_SUPPORT_PATH = "support_path"

class BboxFileHeader:
    COL_PIECE_ID = "#id_piece"
    COL_PIECE_CODE = "#piece_code"
    COL_IMG_NAME = "#img_name"
    COL_PATH_TO_RUNLIST_MASK = "#path_to_runlist_mask"
    COL_CLASS_KEY = "#class_key"
    COL_BBOX_MIN_X = "#bbox.min.x"
    COL_BBOX_MIN_Y = "#bbox.min.y"
    COL_BBOX_MAX_X = "#bbox.max.x"
    COL_BBOX_MAX_Y = "#bbox.max.y"
    COL_ID_DEFECT = "#id_defect"
    COL_ID_CAM = "#id_cam"
    COL_ID_CHANNEL = "#id_channel"
    COL_SOURCE = "#source"
    COL_DEFECT_IS_VALIDATED = "#defect_is_validated"
    COL_SEVERITY = "#severity"
    COL_FAREA = "#fArea"
    COL_FAVGINT = "#fAvgInt"
    COL_FELONG = "#fElong"
    COL_VSCORE = "#vScore"