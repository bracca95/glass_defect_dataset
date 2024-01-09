# inspired by quicktype.io

from __future__ import annotations # ClassName for ClassName in static methods would require 'ClassMethod'

import os
import sys
import json

from functools import reduce
from dataclasses import dataclass, field
from typing import Optional, Union, List, Any, Callable, Iterable, Type, cast

from .tools import Tools, Logger
from ...config.consts import T
from ...config.consts import General as _CG
from ...config.consts import DatasetConst as _CD

def from_bool(x: Any) -> bool:
    Tools.check_instance(x, bool)
    return x

def from_int(x: Any) -> int:
    Tools.check_instance(x, int)
    return x

def from_float(x: Any) -> float:
    Tools.check_instance(x, float)
    return x

def from_str(x: Any) -> str:
    Tools.check_instance(x, str)
    return x

def from_none(x: Any) -> Any:
    Tools.check_instance(x, None)
    return x

def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    Tools.check_instance(x, list)
    return [f(y) for y in x]

def from_union(fs: Iterable[Any], x: Any):
    for f in fs:
        try:
            return f(x)
        except:
            pass
    raise TypeError(f"{x} should be one out of {[type(f.__name__) for f in fs]}")


def to_class(c: Type[T], x: Any) -> dict:
    Tools.check_instance(x, c)
    return cast(Any, x).serialize()


@dataclass
class DatasetConfig:
    dataset_path: str = _CG.DEFAULT_STR
    dataset_type: str = _CG.DEFAULT_STR
    dataset_splits: List[float] = field(default_factory=list)
    normalize: bool = _CG.DEFAULT_BOOL
    crop_size: int = _CG.DEFAULT_INT
    image_size: int = _CG.DEFAULT_INT
    augment_online: Optional[List[str]] = None
    augment_offline: Optional[List[str]] = None
    dataset_mean: Optional[List[float]] = None
    dataset_std: Optional[List[float]] = None
    csv_path: Optional[str] = None
    support_path: Optional[str] = None

    @classmethod
    def deserialize(cls, obj: Any) -> DatasetConfig:
        try:
            Tools.check_instance(obj, dict)
            dataset_path = Tools.validate_path(obj.get(_CD.CONFIG_DATASET_PATH))
        except (FileNotFoundError, ValueError) as fnf:
            Logger.instance().error(fnf.args)
            dataset_path = input("insert dataset path: ")
            while not os.path.exists(dataset_path):
                dataset_path = input("insert dataset path: ")
                
        try:
            dataset_type = from_str(obj.get(_CD.CONFIG_DATASET_TYPE))
            dataset_splits = from_list(lambda x: from_float(x), obj.get(_CD.CONFIG_DATASET_SPLITS))
            normalize = from_bool(obj.get(_CD.CONFIG_NORMALIZE))
            crop_size = from_int(obj.get(_CD.CONFIG_CROP_SIZE))
            image_size = from_int(obj.get(_CD.CONFIG_IMAGE_SIZE))
            augment_online = from_union([lambda x: from_list(from_str, x), from_none], obj.get(_CD.CONFIG_AUGMENT_ONLINE))
            augment_offline = from_union([lambda x: from_list(from_str, x), from_none], obj.get(_CD.CONFIG_AUGMENT_OFFLINE))
            dataset_mean = from_union([lambda x: from_list(from_float, x), from_none], obj.get(_CD.CONFIG_DATASET_MEAN))
            dataset_std = from_union([lambda x: from_list(from_float, x), from_none], obj.get(_CD.CONFIG_DATASET_STD))
            csv_path = from_union([from_str, from_none], obj.get(_CD.CONFIG_CSV_PATH))
            support_path = from_union([from_str, from_none], obj.get(_CD.CONFIG_SUPPORT_PATH))
        except TypeError as te:
            Logger.instance().critical(te.args)
            sys.exit(-1)

        if augment_online is not None:
            if len(augment_online) == 0:
                augment_online = None

        if augment_offline is not None:
            if len(augment_offline) == 0:
                augment_offline = None

        if len(dataset_splits) not in (1, 3):
            raise ValueError("`dataset_splits` must have len == 1 (train/test) or len == 3 (train/val/test)")
        
        if len(dataset_splits) == 3:
            if 1.0-_CG.EPS < reduce(lambda a,b: a+b, dataset_splits) < 1.0+_CG.EPS:
                pass
            else:
                raise ValueError("the sum for dataset_splits must be 1")
            
        if csv_path is not None:
            try:
                csv_path = Tools.validate_path(csv_path)
            except FileNotFoundError as fnf:
                Logger.instance().critical(f"{fnf.args}\nA csv file path has been specified but not found.")

        if support_path is not None:
            try:
                support_path = Tools.validate_path(support_path)
            except FileNotFoundError as fnf:
                Logger.instance().critical(f"{fnf.args}\nA csv file path has been specified but not found.")

        if not any(map(lambda x: x.endswith(".png"), os.listdir(support_path))):
            raise FileNotFoundError(f"The support set directory {support_path} exists but no images are contained")
        
        Logger.instance().info(f"DatasetConfig deserialized: " +
            f"dataset_path: {dataset_path}, dataset_type: {dataset_type}, dataset_splits: {dataset_splits}, " +
            f"normalize: {normalize}, augment_online: {augment_online}, augment_offline: {augment_offline}, " +
            f"dataset mean: {dataset_mean}, dataset_std: {dataset_std}, crop_size: {crop_size}, " +
            f"image_size: {image_size}, csv_path: {csv_path}, support_path: {support_path}"
        )
        
        return DatasetConfig(
            dataset_path, dataset_type, dataset_splits, normalize, crop_size, image_size, augment_online, 
            augment_offline, dataset_mean, dataset_std, csv_path, support_path
        )

    def serialize(self) -> dict:
        result: dict = {}
        
        # if you do not want to write null values, add a field to result if and only if self.field is not None
        result[_CD.CONFIG_DATASET_PATH] = from_str(self.dataset_path)
        result[_CD.CONFIG_DATASET_TYPE] = from_str(self.dataset_type)
        result[_CD.CONFIG_DATASET_SPLITS] = from_list(lambda x: from_float(x), self.dataset_splits)
        result[_CD.CONFIG_NORMALIZE] = from_bool(self.normalize)
        result[_CD.CONFIG_CROP_SIZE] = from_int(self.crop_size)
        result[_CD.CONFIG_IMAGE_SIZE] = from_int(self.image_size)
        result[_CD.CONFIG_AUGMENT_ONLINE] = from_union([lambda x: from_list(from_str, x), from_none], self.augment_online)
        result[_CD.CONFIG_AUGMENT_OFFLINE] = from_union([lambda x: from_list(from_str, x), from_none], self.augment_offline)
        result[_CD.CONFIG_DATASET_MEAN] = from_union([lambda x: from_list(from_float, x), from_none], self.dataset_mean)
        result[_CD.CONFIG_DATASET_STD] = from_union([lambda x: from_list(from_float, x), from_none], self.dataset_std)
        result[_CD.CONFIG_CSV_PATH] = from_union([from_str, from_none], self.csv_path)
        result[_CD.CONFIG_SUPPORT_PATH] = from_union([from_str, from_none], self.support_path)

        Logger.instance().info(f"DatasetConfig serialized: {result}")
        return result


def read_from_json(str_path: str) -> DatasetConfig:
    obj = Tools.read_json(str_path)
    return DatasetConfig.deserialize(obj)

def write_to_json(config: DatasetConfig, directory: str, filename: str) -> None:
    dire = None
    
    try:
        dire = Tools.validate_path(directory)
    except FileNotFoundError as fnf:
        Logger.instance().critical(f"{fnf.args}")
        sys.exit(-1)

    serialized_config = config.serialize()
    with open(os.path.join(dire, filename), "w") as f:
        json_dict = json.dumps(serialized_config, indent=4)
        f.write(json_dict)
