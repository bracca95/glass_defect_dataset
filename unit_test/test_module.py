import os
import sys
import torch
import random
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

from glass_defect_dataset.src.datasets.dataset import DatasetLauncher
from glass_defect_dataset.src.datasets.dataset_utils import DatasetBuilder, YoloDatasetBuilder, FullInferenceBuilder
from glass_defect_dataset.src.utils.config_parser import read_from_json, write_to_json
from glass_defect_dataset.src.utils.tools import Logger

SEED = 1234         # with the first protonet implementation I used 7

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


if __name__=="__main__":
    try:
        config = read_from_json("config/config.json")
    except Exception as e:
        Logger.instance().critical(e.args)
        sys.exit(-1)

    try:
        # check if YOLO
        if "yolo" in config.dataset_type:
            dataset = YoloDatasetBuilder.load_dataset(config)
            sys.exit(0)
        elif "full_inference" in config.dataset_type:
            dataset = FullInferenceBuilder.load_dataset(config)
            dataset.save_exact_defect(margin=5)
            sys.exit(0)
        else:
            dataset = DatasetBuilder.load_dataset(config)
    except ValueError as ve:
        Logger.instance().critical(ve.args)
        sys.exit(-1)

    # compute mean and variance of the dataset if not done yet
    if config.normalize and config.dataset_mean is None and config.dataset_std is None:
        Logger.instance().warning("No mean and std set: computing and storing values.")
        mean, std = DatasetLauncher.compute_mean_std(dataset.train_dataset, config.dataset_type)
        config.dataset_mean = mean.tolist()
        config.dataset_std = std.tolist()
        write_to_json(config, os.getcwd(), "config/config.json")
        sys.exit(0)

    # # instantiate model
    # model = Model().to(_CG.DEVICE)
    
    # # train/test
    # routine = TrainTestExample(model, dataset, subsets_dict)
    # routine.train(config)
    # routine.test(config, model_path)
