import os
import pandas as pd

from PIL import Image
from PIL.Image import Image as PilImgType
from glob import glob
from typing import Optional, List, Tuple
from dataclasses import dataclass

from ...utils.tools import Logger, Tools
from ...utils.config_parser import DatasetConfig
from ....config.consts import PlatePathsDict
from ....config.consts import General as _CG
from ....config.consts import BboxFileHeader as _CH


@dataclass
class Bbox:
    defect_id: int
    defect_class: str
    min_x: int
    max_x: int
    min_y: int
    max_y: int


class BoundingBoxParser:

    CSV_RELATIVE = "2.4_dataset_opt/all/bounding_boxes.csv"
    TMP_COLUMN = "plate_group"

    def __init__(self, dataset_config: DatasetConfig):
        self.dataset_config = dataset_config
        
        # get names of each channel, grouping by plate
        self.all_dataset_images = glob(os.path.join(self.dataset_config.dataset_path, "202*", "*.png"))
        self.plate_name_set = set(map(lambda x: x.rsplit("_", 1)[0], self.all_dataset_images))

        # read csv by filtering classes and returning
        csv_relative_path = self.CSV_RELATIVE
        self._dataset_csv = os.path.join(self.dataset_config.dataset_path, csv_relative_path)
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
        df[_CH.COL_IMG_NAME] = df[_CH.COL_IMG_NAME].apply(
            lambda x: os.path.join(self.dataset_config.dataset_path, f"{os.sep}".join(x.rsplit(os.sep, -1)[-2:]))
        )

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

        return out

    @staticmethod
    def order_by(df: pd.DataFrame, order_by: Optional[str] = None) -> pd.DataFrame:
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
    def group_by_plate_ch1_ch2(_df: pd.DataFrame):
        # group df for same plate (names)
        df = _df.copy()
        df[BoundingBoxParser.TMP_COLUMN] = df[_CH.COL_IMG_NAME].str.extract(r'(.+)_')[0]
        return df.groupby(BoundingBoxParser.TMP_COLUMN).agg(list).reset_index()
    

class SinglePlate:

    def __init__(self, ch_1: str, ch_2: str):
        self.ch_1 = ch_1
        self.ch_2 = ch_2

        self.defects: List[Bbox] = list()

    def tolist(self) -> List[str]:
        return [self.ch_1, self.ch_2]
    
    def to_platepaths(self) -> PlatePathsDict:
        return { "ch_1": self.ch_1, "ch_2": self.ch_2 }
    
    def read_full_img(self, mode: str="L") -> Tuple[PilImgType, PilImgType]:
        img_1 = Image.open(self.ch_1).convert(mode)
        img_2 = Image.open(self.ch_2).convert(mode)

        return img_1, img_2

    def locate_on_plate(self, df_grouped: pd.DataFrame, columns: List[str], filt: Optional[List[str]]) -> List[Bbox]:        
        # look for all the defects of that plate in the df
        lookup_df = df_grouped[df_grouped[_CH.COL_IMG_NAME].apply(lambda x: all(item in x for item in self.tolist()))]
        if lookup_df.empty: return list()
        lookup_df = lookup_df.explode(list(columns), ignore_index=True).drop(columns=[BoundingBoxParser.TMP_COLUMN])

        # wrap
        defects: List[Bbox] = list()
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

            defects.append(Bbox(defect_id, defect_class, bbox_min_x, bbox_max_x, bbox_min_y, bbox_max_y))

        return defects

class DefectOptDataset:

    FULL_IMG_LIST = "test_plates.txt"

    def __init__(self, dataset_config: DatasetConfig):
        self.csv = BoundingBoxParser(dataset_config)
        self.full_img_set = self._read_plate_file(Tools.validate_path(self.FULL_IMG_LIST))
        self._save_exact_defect()

    def _save_exact_defect(self, margin: int=5):
        df = self.csv.group_by_plate_ch1_ch2(self.csv._df)

        for plate in self.full_img_set:
            plate.defects = plate.locate_on_plate(df, list(self.csv._df.columns), filt=None)
            
            img_1 = Image.open(plate.ch_1).convert("L")
            img_2 = Image.open(plate.ch_2).convert("L")

            for defect in plate.defects:
                if margin == 0:
                    defect_coords = (defect.min_x, defect.min_y, defect.max_x, defect.max_y)
                else:
                    cx = (defect.max_x + defect.min_x) // 2
                    cy = (defect.max_y + defect.min_y) // 2
                    side = max((defect.max_x - defect.min_x), (defect.max_y - defect.min_y))
                    min_x = cx - (side // 2) - margin
                    min_y = cy - (side // 2) - margin
                    max_x = cx + (side // 2) + margin
                    max_y = cy + (side // 2) + margin
                    
                    defect_coords = (min_x, min_y, max_x, max_y)

                defect_1 = img_1.crop(defect_coords)
                defect_2 = img_2.crop(defect_coords)

                out_dir = os.path.join(os.getcwd(), "output")
                defect_vid_1 = f"{defect.defect_class}_did_{defect.defect_id}_vid_1.png"
                defect_vid_2 = f"{defect.defect_class}_did_{defect.defect_id}_vid_2.png"

                try:
                    defect_1.save(os.path.join(out_dir, defect_vid_1))
                    defect_2.save(os.path.join(out_dir, defect_vid_2))
                except SystemError:
                    Logger.instance().error(f"There is an error in the bounding box, check values: {defect}")
                except AttributeError:
                    Logger.instance().error(f"There is an error in the bounding box, check values: {defect}")
    
    def _read_plate_file(self, path_to_txt: Optional[str]) -> set[SinglePlate]:
        msg = f"Using all plates: the path {path_to_txt} does not exist. Create a `$PROJ/test_plates.txt` file"

        # return all the plates if no txt file is specified
        if path_to_txt is None:
            Logger.instance().warning(f"Using all plates in test.")
            return set([SinglePlate(f"{name}_1.png", f"{name}_2.png") for name in self.csv.plate_name_set])
        
        # return all the plates if no txt file is specified
        try:
            path_to_txt = Tools.validate_path(path_to_txt)
        except ValueError as ve:
            Logger.instance().warning(f"{ve.args}\n{msg}")
            return set([SinglePlate(f"{name}_1.png", f"{name}_2.png") for name in self.csv.plate_name_set])
        except FileNotFoundError as fnf:
            Logger.instance().warning(f"{fnf.args}\n{msg}")
            return set([SinglePlate(f"{name}_1.png", f"{name}_2.png") for name in self.csv.plate_name_set])
        
        with open(path_to_txt, "r") as f:
            lines = [p.strip().replace("_1.png", "").replace("_2.png", "").replace(",", "").replace(";", "") for p in f]

        filter_plate_names = set(lines) & self.csv.plate_name_set
        plates = set([SinglePlate(f"{name}_1.png", f"{name}_2.png") for name in filter_plate_names])
        
        Logger.instance().warning(f"Filtering plates: {filter_plate_names}")
        return plates