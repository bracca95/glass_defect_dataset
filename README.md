# Baseline Project for Defect Views dataset

## Dataset
Your own dataset can be built by implementing the abstract class `DatasetWrapper` in `src.dataset.dataset.py`

## Dataset Config
Edit the `config/config.json` file to start

```
"dataset_path": string,
"dataset_type": {`opt6`, `opt_bckg`, `opt_double`, `opt_double_inference`, `binary`, `qplusv1`, `qplusv2`, `qplus_double`, `omniglot`, `miniimagenet`, `opt_yolo_train`, `opt_yolo_test`, `cub`, `cifar_fs`, `celeba`, `full_inference`},
"dataset_splits": List[float] (1 for train/test (e.g. [0.8]), 3 for train/val/test),
"normalize": bool,
"crop_size": int,
"image_size": int (after crop, reshape can be applied),
"augment_online": Optional[List[str]] (classes for online augmentation),
"augment_offline": Optional[List[str]] (classes for offline augmentation),
"dataset_mean": Optional[List[float]] (Grayscale/RGB),
"dataset_std": Optional[List[float]] (Grayscale/RGB),
"csv_path": Optional[str],
"support_path": Optional[str]
```

If `dataset_mean` and `dataset_std` are set to null, the program will compute them and then it will quit the execution.
Run the program again to train your model.