from __future__ import annotations

import os
import math
import pandas as pd

from PIL import Image
from PIL.Image import Image as PilImgType
from glob import glob
from functools import reduce
from itertools import groupby
from typing import Optional, Union, List, Tuple
from dataclasses import dataclass
from torch.utils.data import Dataset as TorchDataset
from torchvision.transforms import transforms

from ...utils.tools import Logger, Tools
from ...utils.config_parser import DatasetConfig
from ....config.consts import PlatePathsDict
from ....config.consts import General as _CG
from ....config.consts import BboxFileHeader as _CH


class SaveDefects:
    active = False
    p_size = 640
    p_stride = 600
    out_dir = "/media/lorenzo/M/datasets/dmx/dmx_2c/november/unclass_other"
    csv_relative_path = "2.4_dataset_opt/unclassified/bounding_boxes.csv"
    filt_class = ["bubble_small", "dirt_td", "inclusion", "point_td", "scratch", "det_failed", "no_defect", "unclassified"]


@dataclass
class Bbox:
    defect_class: str
    min_x: int
    max_x: int
    min_y: int
    max_y: int


class Patch:

    def __init__(self, plate_paths: PlatePathsDict, start_w: int, start_h: int, w: int, h: int):
        self.plate_paths = plate_paths
        self.start_w = start_w
        self.start_h = start_h
        self.w = w
        self.h = h
        
        self.defects: Optional[List[Bbox]] = None

    def __eq__(self, __value: Patch) -> bool:
        if self.plate_paths == __value.plate_paths \
            and self.start_w == __value.start_w \
            and self.start_h == __value.start_h \
            and self.w == __value.w \
            and self.h == __value.h:
            return True
        
        return False

    def map_defect_locally(self, abs_bbox: Bbox):
        """Map defect on a single patch

        Starting from the world frame location, convert it to the body frame. Since there are two channels, which have
        slightly different bounding boxes (depending on the intensity for each channel), wrap them to assign one
        bound box for each two-channel image.

        Args:
            abs_bbox (Bbox): the values read from the csv file
        """

        rel_bbox_min_x = abs_bbox.min_x - self.start_w
        rel_bbox_max_x = rel_bbox_min_x + (abs_bbox.max_x - abs_bbox.min_x)
        rel_bbox_min_y = abs_bbox.min_y - self.start_h
        rel_bbox_max_y = rel_bbox_min_y + (abs_bbox.max_y - abs_bbox.min_y)

        relative_bbox = Bbox(abs_bbox.defect_class, rel_bbox_min_x, rel_bbox_max_x, rel_bbox_min_y, rel_bbox_max_y)
        
        if self.defects is None:
            self.defects = [relative_bbox]
            return
        
        self.defects.append(relative_bbox)


class SinglePlate:
    """Single glass plate class"""

    PATCH_SIZE = 60 if not SaveDefects.active else SaveDefects.p_size
    PATCH_STRIDE = 50 if not SaveDefects.active else SaveDefects.p_stride
    UPSCALE = 640

    idx_to_label = {
        0: "bubble"
    }

    label_to_idx = Tools.invert_dict(idx_to_label)

    def __init__(self, ch_1: str, ch_2: str):
        self.ch_1 = ch_1
        self.ch_2 = ch_2

        self.patch_list: List[Patch] = list()

    def tolist(self) -> List[str]:
        return [self.ch_1, self.ch_2]
    
    def to_platepaths(self) -> PlatePathsDict:
        return { "ch_1": self.ch_1, "ch_2": self.ch_2 }
    
    def read_full_img(self, mode: str="L") -> Tuple[PilImgType, PilImgType]:
        img_1 = Image.open(self.ch_1).convert(mode)
        img_2 = Image.open(self.ch_2).convert(mode)

        return img_1, img_2

    def create_patch_list(self, patch_w: int, patch_h: int, stride: int) -> List[Patch]:
        """Create a list of patches for a single glass plate

        Use a sliding window approach to create fixed size image patch where defects are mapped from the world frame
        (absolute values found in the csv) to the body frame (local patch). The full image is open to check its size.
        There might be faster method to infer it. This method collects ALL the possible patches for a single plate,
        independentely of any filters or anomalies.

        Args:
            patch_w (int): desired patch width (columns)
            patch_h (int): desired patch height (rows)
            stride (int): stride for the sliding window

        Returns:
            List[Patch]
        """

        img_w, img_h = Image.open(self.ch_1).size
        return self.sliding_window(self, img_w, img_h, patch_w, patch_h, stride)
    
    def locate_defects(self, lookup_df: pd.DataFrame, filt: Optional[List[str]], inference: bool=False):
        """Locate defects in the patch

        Starting from the world frame location, convert it to the body frame. Since there are two channels, which have
        slightly different bounding boxes (depending on the intensity for each channel), wrap them to assign one
        bound box for each two-channel image. Do this for every patch.

        Args:
            lookup_df (pd.DataFrame): the dataframe grouped by defects for that specific plate.
            filt (Optional[List[str]]): classes to select. If None, use all defect classes
            inference (bool): should be True when testing yolo so that you won't lose any patches that contain defects
        """

        if len(self.patch_list) == 0:
            msg = f"Trying to locate defects, but patches have not been initialized yet."
            Logger.instance().error(msg)
            raise ValueError(msg)

        # wrap
        for _, group in lookup_df.groupby(_CH.COL_ID_DEFECT):
            defect_id = group[_CH.COL_ID_DEFECT].tolist()[0]
            defect_class = group[_CH.COL_CLASS_KEY].tolist()[0]

            # skip
            if defect_class is not None and filt is not None:
                if defect_class not in filt: 
                    Logger.instance().info(f"skipping defect class {defect_class}: not included in {filt}")
                    continue
            
            bbox_min_x = min(group[_CH.COL_BBOX_MIN_X])
            bbox_max_x = max(group[_CH.COL_BBOX_MAX_X])
            bbox_min_y = min(group[_CH.COL_BBOX_MIN_Y])
            bbox_max_y = max(group[_CH.COL_BBOX_MAX_Y])

            if bbox_min_x == bbox_max_x:
                Logger.instance().warning(f"min/max x overlap in defect {defect_id}: increasing bb width")
                bbox_min_x -= 1
                bbox_max_x += 1

            if bbox_min_y == bbox_max_y:
                Logger.instance().warning(f"min/max y overlap in defect {defect_id}: increasing bb height")
                bbox_min_y -= 1
                bbox_max_y += 1

            # check if there is a defect in a patch
            for patch in self.patch_list:
                if bbox_min_x > patch.start_w and bbox_min_y > patch.start_h \
                and bbox_max_x < patch.start_w + patch.w and bbox_max_y < patch.start_h + patch.h \
                and defect_class in filt:
                    abs_bbox = Bbox(defect_class, bbox_min_x, bbox_max_x, bbox_min_y, bbox_max_y)
                    patch.map_defect_locally(abs_bbox)

                    if SaveDefects.active:
                        self.__save_exact_defect(defect_id, patch, 5)
                    
                    # during inference every defect that appears more than once must be included
                    if not inference:
                        break # during training, avoid putting it in both train and test dataset
    
    @staticmethod
    def sliding_window(plate: SinglePlate, img_w: int, img_h: int, patch_w: int, patch_h: int, stride: int) -> List[Patch]:
        """Sliding window over a plate

        Args:
            plate (SinglePlate): the image for which I want to create patches
            img_w (int): SinglePlate's original width
            img_h (int): SinglePlate's original height
            patch_w (int): desired patch width
            patch_h (int): desired patch height
            stride (int): desired stride
        """
        
        # the image size is reduced. Consider enhancing the method by padding
        patch_list: List[Patch] = list()
        for i in range(0, img_h - patch_h + 1, stride):
            for j in range(0, img_w - patch_w + 1, stride):
                patch = Patch(plate.to_platepaths(), j, i, patch_w, patch_h)
                patch_list.append(patch)
            
            # include last column patch for each line
            patch = Patch(plate.to_platepaths(), img_w - patch_w, i, patch_w, patch_h)
            patch_list.append(patch)

        # include last row patches (last column not included)
        for j in range(0, img_w - patch_w + 1, stride):
            patch = Patch(plate.to_platepaths(), j, img_h - patch_h, patch_w, patch_h)
            patch_list.append(patch)

        # include last column (bottom right)
        patch = Patch(plate.to_platepaths(), img_w - patch_w, img_h - patch_h, patch_w, patch_h)
        patch_list.append(patch)

        return patch_list
    
    @staticmethod
    def __debug_save_image_patch(patch: Patch):
        # call this in SinglePlate::locate_defects
        from PIL import ImageDraw
        img = Image.open(patch.plate_paths["ch_1"]).convert("RGB")
        crop = img.crop((patch.start_w, patch.start_h, patch.start_w + patch.w, patch.start_h + patch.h))
        draw = ImageDraw.Draw(crop)
        draw.rectangle([patch.defects[0].min_x, patch.defects[0].min_y, patch.defects[0].max_x, patch.defects[0].max_y], (255, 0, 0))
        crop.save("output/patch.png")

    @staticmethod
    def __save_exact_defect(defect_id: int, patch: Patch, margin: int):
        # call this in SinglePlate::locate_defects
        img_1 = Image.open(patch.plate_paths["ch_1"]).convert("L")
        img_2 = Image.open(patch.plate_paths["ch_2"]).convert("L")

        patch_coords = (patch.start_w, patch.start_h, patch.start_w + patch.w, patch.start_h + patch.h)

        if margin == 0:
            defect_coords = (patch.defects[0].min_x, patch.defects[0].min_y, patch.defects[0].max_x, patch.defects[0].max_y)
        else:
            cx = (patch.defects[0].max_x + patch.defects[0].min_x) // 2
            cy = (patch.defects[0].max_y + patch.defects[0].min_y) // 2
            side = max((patch.defects[0].max_x - patch.defects[0].min_x), (patch.defects[0].max_y - patch.defects[0].min_y))
            min_x = cx - (side // 2) - margin
            min_y = cy - (side // 2) - margin
            max_x = cx + (side // 2) + margin
            max_y = cy + (side // 2) + margin

            if min_x < 0 or min_y < 0 or max_x > patch.w or max_y > patch.h:
                Logger.instance().warning(f"cannot fit defect into patch")
                return
            
            defect_coords = (min_x, min_y, max_x, max_y)

        patch_1 = img_1.crop(patch_coords)
        patch_2 = img_2.crop(patch_coords)

        defect_1 = patch_1.crop(defect_coords)
        defect_2 = patch_2.crop(defect_coords)

        out_dir = SaveDefects.out_dir
        defect_vid_1 = f"{patch.defects[0].defect_class}_did_{defect_id}_vid_1.png"
        defect_vid_2 = f"{patch.defects[0].defect_class}_did_{defect_id}_vid_2.png"

        try:
            defect_1.save(os.path.join(out_dir, defect_vid_1))
            defect_2.save(os.path.join(out_dir, defect_vid_2))
        except SystemError:
            Logger.instance().error(f"There is an error in the bounding box, check values: {patch.defects[0]}")
        except AttributeError:
            Logger.instance().error(f"There is an error in the bounding box, check values: {patch.defects[0]}")


class GlassPlate:

    def __init__(self, dataset_config: DatasetConfig):
        self.dataset_config = dataset_config
        self._dataset_opt_main = "/media/lorenzo/M/datasets/dmx/dataset_opt"
        
        # get names of each channel, grouping by plate
        self.all_dataset_images = glob(os.path.join(self._dataset_opt_main, "202*", "*.png"))
        self.plate_name_set = set(map(lambda x: x.rsplit("_", 1)[0], self.all_dataset_images))

        # read csv by filtering classes and returning
        csv_relative_path = "2.4_dataset_opt/riclassified/bounding_boxes.csv"
        if SaveDefects.active:
            csv_relative_path = SaveDefects.csv_relative_path
        self._dataset_csv = os.path.join(self._dataset_opt_main, csv_relative_path)
        self._df = self._parse_csv(self.all_dataset_images, filt=None)

    def _parse_csv(self, available_images: List[str], filt: Optional[List[str]]=None) -> pd.DataFrame:
        """Read bounding boxes csv (parser)
        
        Reads the bounding_boxes.csv file correctly:
            1. separator is ";",
            2. index_col is broken,
            3. remove NaN entries from image names and bboxes
            4. bboxes from float to int
            5. replace full path
        At the end, the df is returned ordered following the `filt` argument, if specified.

        Args:
            filt (Optional[List[str]]): list with classes of defects that must be accounted

        Returns:
            pd.DataFrame
        """
        
        df = pd.read_csv(self._dataset_csv, sep=";", index_col=False)

        # check if all the required columns exist
        col_values = [getattr(_CH, attr) for attr in dir(_CH) if attr.startswith("COL")]
        if not set(col_values) == set(df.columns):
            raise ValueError(f"Some columns are not present. Check:\ndf: {df.columns}\nyours: {col_values}")

        # remove wrong values (NaN)
        to_int_cols = [_CH.COL_BBOX_MIN_X, _CH.COL_BBOX_MAX_X, _CH.COL_BBOX_MIN_Y, _CH.COL_BBOX_MAX_Y]
        not_be_nan_cols = [_CH.COL_IMG_NAME] + to_int_cols
        df = df.dropna(subset=not_be_nan_cols)

        # bbox to int
        df[to_int_cols] = df[to_int_cols].astype(int)

        # replace path
        df[_CH.COL_IMG_NAME] = df[_CH.COL_IMG_NAME].apply(lambda x: os.path.join(self._dataset_opt_main, f"{os.sep}".join(x.rsplit(os.sep, -1)[-2:])))

        # class filters
        if filt is not None:
            defect_classes = list(self.get_defect_class(df))
            selected_columns = set()
            for col in set(filt):
                # check if wrong names were inserted in the config.json file
                if col not in defect_classes:
                    Logger.instance().warning(f"No defect named {col}")
                else:    
                    selected_columns.add(col)

            df = df.loc[df[_CH.COL_CLASS_KEY].isin(selected_columns)]

        # check if df contains entries for filenames (image plates) that are not available on the device
        missing = set(df[_CH.COL_IMG_NAME].to_list()) - set(available_images)
        if not missing == set():
            Logger.instance().warning(f"The following plates will be removed from df since not available: {missing}")
            df = df[~df[_CH.COL_IMG_NAME].isin(missing)]

        return df

    def get_defect_class(self, df: pd.DataFrame):
        return set(df[_CH.COL_CLASS_KEY].unique())

    def n_defect_per_class(self, patches_w_defects: List[Patch], filt: Optional[List[str]]) -> dict:
        filter_patches = patches_w_defects.copy()
        cls_filt = set(self._df[_CH.COL_CLASS_KEY].tolist())

        # check if at least one element of the one specified (filt) is in the defect list for each pat
        if filt is not None:
            cls_filt = set(filt)
            filter_patches = list(filter(lambda x: len(set([d.defect_class for d in x.defects]) & (cls_filt)) > 0, patches_w_defects))
        
        # init defect dictionary
        numel_defect = { c: 0 for c in cls_filt }
        for patch in filter_patches:
            for d in patch.defects:
                for f in cls_filt:
                    if d.defect_class == f:
                        numel_defect[f] += 1

        return numel_defect

    def get_one_view_per_channel(self, df: pd.DataFrame, order_by: Optional[List[str]]=None) -> Optional[pd.DataFrame]:
        """Return one view for each channel

        Sometimes multiple views are linked to the same image channel. REMOVE the ones that exceeds.
        This method also orders the outcome df, if specified.

        Args:
            df (pd.DataFrame): may apply on differently filtered dataframes, not necessarily the main of this class
            order_by (Optional[list[str]]): if you want to return a csv with a different order.

        Returns:
            Optional[pd.DataFrame]
        """
        
        # locate the 3rd, 4th, ... occurrence (view) for a defect and remove from db
        # https://stackoverflow.com/a/70168878
        out = df[df.groupby(_CH.COL_ID_DEFECT).cumcount().le(1)]

        return self.order_by(out, order_by)

    @staticmethod
    def order_by(df: pd.DataFrame, order_by: Optional[List[str]] = None) -> pd.DataFrame:
        """Order by one/more different column/s
    
        Args:
            df (pd.DataFrame): the input csv
            order_by (lis[str]): the selected columns

        Returns:
            pd.DataFame
            
        """

        if order_by is None or len(order_by) == 0:
            Logger.instance().debug("Order by: nothing to do!")
            return df
        
        if any(list(map(lambda x: x not in df.columns, order_by))):
            Logger.instance().error(f"sort by: {order_by}")
            raise ValueError("Check config file: some columns may not be present")
            
        Logger.instance().debug(f"sort by: {order_by}")
        return df.sort_values(order_by, ascending=[True] * len(order_by))

    @staticmethod
    def sort_by_plate_filename(patch_list: List[Patch]) -> List[SinglePlate]:
        objs = patch_list.copy()
        objs.sort(key=lambda x: x.plate_paths["ch_1"])
        
        plate_list: List[SinglePlate] = list()
        for key, group in groupby(objs, key=lambda x: x.plate_paths["ch_1"]):
            plate_id = key.rsplit("_1.png")[0]
            plate = SinglePlate(f"{plate_id}_1.png", f"{plate_id}_2.png")
            plate.patch_list = list(group)
            plate_list.append(plate)

        return plate_list
    
    @staticmethod
    def group_by_plate_ch1_ch2(_df: pd.DataFrame):
        # group df for same plate (names)
        df = _df.copy()
        df["plate_group"] = df[_CH.COL_IMG_NAME].str.extract(r'(.+)_')[0]
        return df.groupby("plate_group").agg(list).reset_index()
    

class GlassPlateTrainYolo(GlassPlate):

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)
        self._check_existance()

    def _check_existance(self):
        subdirs = set(os.listdir(self.dataset_config.dataset_path))
        if set(["train", "test", "val"]) <= subdirs:
            if "data.yaml" not in subdirs:
                msg = f"data.yaml not in {self.dataset_config.dataset_path}: add it there and run the program again."
                Logger.instance().critical(msg)
                raise FileNotFoundError(msg)

            # should also check if those directories contain images tbf...
            Logger.instance().debug(f"Dataset images found for train/test yolo :)")
            return

        ## if images have not been stored yet
        Logger.instance().debug(f"Storing images at {self.dataset_config.dataset_path}. May take some time..")

        # find defects in each plate
        patches_with_defects = self._locate_defects_all_plates(self.plate_name_set)
    
        if not SaveDefects.active:
            # split train/val/test
            train_list, val_list, test_list = self._train_test_split(patches_with_defects)

            # save yolo format patches
            self._save_patches_yolo_format(train_list, "train")
            self._save_patches_yolo_format(val_list, "val")
            self._save_patches_yolo_format(test_list, "test")


    def _locate_defects_all_plates(self, plate_name_set: set[str]) -> List[SinglePlate]:
        """Call locate defect for each plate

        This method is a pain in the arse because when you instantiate SinglePlate, the sliding window method that
        creates patches for the whole plate is automatically called. It wouldn't be so bad, it could be called later,
        but the main problem is that I would eventually need to store a collection of plates with the corresponding
        `patch_list`: this would mean storing too much information and I have already experienced a sort of program 
        suicide due to out-of-memory. Hence, the only way to make things faster is to instantiate a new plate for 
        every name and append only the defective patches when needed. The memory will be eventually cleared at the end 
        of each loop iteration (with respect to the whole plate, I mean).
        Since this is used only in the YOLO training part, we can filter out patches that do not contain defects, so
        the final list `SinglePlate::patch_list` will embed only defective patches.
        
        Args:
            plate_name_set (set[str]): names of the images without channel info ("_1.png", "_2.png")

        Returns:
            List[Patch]
        """
        
        df = GlassPlate.group_by_plate_ch1_ch2(self._df)
        
        tot_defects = 0
        plates_with_defects: List[SinglePlate] = list()
        for plate_name in plate_name_set:
            # create a patch list (with sliding window for each plate)
            plate = SinglePlate(f"{plate_name}_1.png", f"{plate_name}_2.png")
            plate.patch_list =  plate.create_patch_list(
                patch_w=SinglePlate.PATCH_SIZE,
                patch_h=SinglePlate.PATCH_SIZE,
                stride=SinglePlate.PATCH_STRIDE
            )
            
            # look for all the defects of that plate in the df
            lookup_df = df[df[_CH.COL_IMG_NAME].apply(lambda x: all(item in x for item in plate.tolist()))]
            if lookup_df.empty: continue
            lookup_df = lookup_df.explode(list(self._df.columns), ignore_index=True).drop(columns=["plate_group"])
            
            # locate the defects in the patch list for the current plate
            filters = SaveDefects.filt_class if SaveDefects.active else list(SinglePlate.label_to_idx.keys())
            plate.locate_defects(lookup_df, filt=filters)
            
            # override: filter out the original patch list and keep only the patches that actually contain defects
            plate.patch_list = list(filter(lambda x: x.defects is not None, plate.patch_list))
            if len(plate.patch_list) > 0:
                tot_defects += len(plate.patch_list)
                plates_with_defects.append(plate)

        Logger.instance().debug(f"{tot_defects} patches contain defects ({list(SinglePlate.label_to_idx.keys())}).")
        return plates_with_defects

    def _train_test_split(self, defective_plates: List[SinglePlate]) -> Tuple[List[SinglePlate], List[SinglePlate], List[SinglePlate]]:
        """Split in train/val/test for YOLO
        
        Each split contain the exact number (in percentage) of defects defined in the config.json file, for each defect
        class. The list is finally ordered because reading random patches would means to load the whole image for
        every single patch (too slow). Thus, plate image is read once, then all the defective patches for each plate
        are cropped.

        Args:
            defective_plates (List[SinglePlate]): this method must be called after defect localization

        Returns:
            train, val, test (Tuple[List[SinglePlate], List[SinglePlate], List[SinglePlate]])
        """

        # check settings
        if len(self.dataset_config.dataset_splits) < 3:
            raise ValueError(f"YOLO need train/val/test splits: set config to have three splits.")
        
        # check every patch has defects
        defective_patches: List[Patch] = list()
        [defective_patches.extend(p.patch_list) for p in defective_plates]

        if any(map(lambda x: x.defects is None, defective_patches)):
            raise ValueError("Must provide patches with defects.")
        
        ratios = self.dataset_config.dataset_splits.copy()
        filt = list(SinglePlate.idx_to_label.values())
        n_defect: dict = self.n_defect_per_class(defective_patches, filt=filt)
        
        ds_len = reduce(lambda a, b: a+b, list(n_defect.values()))
        splits = { k: [
                int(v * ratios[0]),
                int(v * ratios[1]),
                v-(int(v * ratios[0]) + int(v * ratios[1]))
            ] for k, v in n_defect.items() 
        }

        train_list_defect: List[Patch] = list()
        val_list_defect: List[Patch] = list()
        test_list_defect: List[Patch] = list()

        for k, v in splits.items():
            count_val = 0
            count_test = 0
            for patch in defective_patches:
                if any(map(lambda x: x.defect_class == k, patch.defects)):
                    if count_val < v[1]:
                        val_list_defect.append(patch)
                        count_val += 1
                    elif count_test < v[2]:
                        test_list_defect.append(patch)
                        count_test += 1
                    else:
                        train_list_defect.append(patch)

        ord_train_list = self.sort_by_plate_filename(train_list_defect)
        ord_val_list = self.sort_by_plate_filename(val_list_defect)
        ord_test_list = self.sort_by_plate_filename(test_list_defect)

        return ord_train_list, ord_val_list, ord_test_list

    def _save_patches_yolo_format(self, plate_list: List[SinglePlate], split: str):
        """Save for YOLO training

        In this case, we consider that all the patches have defects.

        Args:
            plate_idx (int)
            plate_patch_list (List[Patch]): `SinglePlate::patch_list_defects`
            split (str): "train", "val", "test"
        """

        img_size = SinglePlate.UPSCALE
        image_folder_path = os.path.join(self.dataset_config.dataset_path, split, "images")
        label_folder_path = os.path.join(self.dataset_config.dataset_path, split, "labels")

        if not os.path.exists(image_folder_path): os.makedirs(image_folder_path)
        if not os.path.exists(label_folder_path): os.makedirs(label_folder_path)

        for plate in plate_list:
            img_1, img_2 = plate.read_full_img()
            plate_id = os.path.basename(plate.ch_1.split("_1.png")[0])

            for patch_idx, patch in enumerate(plate.patch_list):
                patch_basename = f"plate_{plate_id}_patch_{patch_idx}"
                patch_filename = os.path.join(image_folder_path, f"{patch_basename}.png")
                label_filename = os.path.join(label_folder_path, f"{patch_basename}.txt")
                
                # save image
                patch_pil = GlassPlateTrainYolo.patch_to_pil(img_1, img_2, patch, img_size)
                patch_pil.save(patch_filename)

                # writing annotations
                GlassPlateTrainYolo.write_patch_annotations(patch, label_filename)

    @staticmethod
    def patch_to_pil(img_1: PilImgType, img_2:PilImgType, patch: Patch, img_size: int, mode: str="L") -> PilImgType:
        """Return a patch as a PIL image type

        Args:
            img_1 (PilImgType): channel 1 (tb) for full plate
            img_2 (PilImgType): channel 2 (td) for full plate
            patch (Patch)
            img_size (int): used for resizing to YOLO default format

        Returns:
            PilImgType
        """
        
        # image crop
        crop_1 = img_1.crop((patch.start_w, patch.start_h, patch.start_w + patch.w, patch.start_h + patch.h))
        crop_2 = img_2.crop((patch.start_w, patch.start_h, patch.start_w + patch.w, patch.start_h + patch.h))
        
        if mode == "L":
            ## alpha channel
            img_merge = Image.new("LA", crop_1.size)
            img_merge.paste(crop_1, (0, 0))
            img_merge.paste(crop_2, (0, 0), crop_2)
        else:
            ## RGB
            img_merge = Image.merge("RGB", (crop_1, crop_2, crop_2))

        return img_merge.resize((img_size, img_size), resample=Image.BILINEAR)
    
    @staticmethod
    def write_patch_annotations(patch: Patch, label_filename: str):
        """Write annotations in YOLO format

        Args:
            patch (Patch)
            label_filename (str)
        """

        with open(label_filename, "a") as f:
            for didx, defect in enumerate(patch.defects):
                x = float((defect.max_x + defect.min_x) / 2) / float(SinglePlate.PATCH_SIZE)
                y = float((defect.max_y + defect.min_y) / 2) / float(SinglePlate.PATCH_SIZE)
                w = float((defect.max_x - defect.min_x) / float(SinglePlate.PATCH_SIZE))
                h = float((defect.max_y - defect.min_y) / float(SinglePlate.PATCH_SIZE))
                line = f"{SinglePlate.label_to_idx[defect.defect_class]} {x} {y} {w} {h}"
                if didx == 0:
                    f.writelines(line)
                else:
                    f.writelines(f"\n{line}")

    @staticmethod
    def read_annotations_back(label_filename: str) -> Optional[List[Bbox]]:
        defects = []
        with open(label_filename, "r") as f:  
            for line in f:
                dc, x, y, w, h = line.strip().rsplit(" ", -1)
                dc, x, y, w, h = (int(dc), float(x), float(y), float(w), float(h))
                left = int(math.floor((x - w/2) * SinglePlate.PATCH_SIZE))
                upper = int(math.floor((y - h/2) * SinglePlate.PATCH_SIZE))
                right = int(math.ceil((x + w/2) * SinglePlate.PATCH_SIZE))
                bottom = int(math.ceil((y + h/2) * SinglePlate.PATCH_SIZE))
                bbox = Bbox(SinglePlate.idx_to_label[dc], left, right, upper, bottom)
                defects.append(bbox)
        return defects if len(defects) > 0 else None  


class GlassPlateTestYolo(GlassPlate):

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)
    
        self.filtered_plates = self._read_plate_file("test_plates.txt")

    def _read_plate_file(self, path_to_txt: Optional[str]) -> List[SinglePlate]:
        msg = f"Using all plates: the path {path_to_txt} does not exist. Create a `$PROJ/test_plates.txt` file"

        # return all the plates if no txt file is specified
        if path_to_txt is None:
            Logger.instance().warning(f"Using all plates in test.")
            return [SinglePlate(f"{name}_1.png", f"{name}_2.png") for name in self.plate_name_set]
        
        # return all the plates if no txt file is specified
        try:
            path_to_txt = Tools.validate_path(path_to_txt)
        except ValueError as ve:
            Logger.instance().warning(f"{ve.args}\n{msg}")
            return [SinglePlate(f"{name}_1.png", f"{name}_2.png") for name in self.plate_name_set]
        except FileNotFoundError as fnf:
            Logger.instance().warning(f"{fnf.args}\n{msg}")
            return [SinglePlate(f"{name}_1.png", f"{name}_2.png") for name in self.plate_name_set]
        
        with open(path_to_txt, "r") as f:
            lines = [p.strip().replace("_1.png", "").replace("_2.png", "").replace(",", "").replace(";", "") for p in f]

        filter_plate_names = set(lines) & self.plate_name_set
        plates = [SinglePlate(f"{name}_1.png", f"{name}_2.png") for name in filter_plate_names]
        
        Logger.instance().warning(f"Filtering plates: {filter_plate_names}")
        return plates

    def analyze_plate(self, plate: SinglePlate, batch_size: int=16):
        if self.filtered_plates == set():
            Logger.instance().debug("No plates in `$PROJ/test_plates.txt`. Add plates or remove the file to use all.")
            return

        df = GlassPlate.group_by_plate_ch1_ch2(self._df)
        
        # store all the patches in memory for a plate
        plate.patch_list =  plate.create_patch_list(
            patch_w=SinglePlate.PATCH_SIZE,
            patch_h=SinglePlate.PATCH_SIZE,
            stride=SinglePlate.PATCH_STRIDE
        )
        
        # look up defect on the current plate
        lookup_df = df[df[_CH.COL_IMG_NAME].apply(lambda x: all(item in x for item in plate.tolist()))]
        if lookup_df.empty: return
        lookup_df = lookup_df.explode(list(self._df.columns), ignore_index=True).drop(columns=["plate_group"])
        
        # store also info about the defective patches on that plate and override
        plate.locate_defects(lookup_df, list(SinglePlate.label_to_idx.keys()), inference=True)
        
        # plate.patch_list = list(filter(lambda x: x.defects is not None, plate.patch_list))
        # if len(plate.patch_list) == 0:
        #     return
        
        img_1, img_2 = plate.read_full_img()
        img_size = SinglePlate.UPSCALE

        batch_patch = []
        batch_patch_pil = []

        # use yield to return ALL the patches (list) for a plate. YOLO will have to manage one plate at a time
        for patch in plate.patch_list:
            img_pil = GlassPlateTrainYolo.patch_to_pil(img_1, img_2, patch, img_size)
            batch_patch.append(patch)
            batch_patch_pil.append(img_pil)
            if len(batch_patch) == batch_size:
                yield batch_patch, batch_patch_pil
                batch_patch, batch_patch_pil = [], []

        # If there are any remaining images in batch_images after the loop, yield them
        if batch_patch:
            yield batch_patch, batch_patch_pil