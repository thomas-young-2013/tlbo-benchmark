## Hyperparameter tuing (HPT) group in DAIM
See descriptions under docs for more details!

## How to create metadata for tuning Xgboost
Just type in your terminal:
```
cd tl-benchmark
python benchmark/tuner/tuning_xgb.py --dataset <dataset name>
```

## Add NEW dataset to this benchmark
1. the interface is definede in `benchmark.utils.load_xgb_dataset.py`.
2. you need to add the source file for reading the dataset to the folder `benchmark.utils.dataset_scripts`.
3. a tutorial example given is dataset `fall_detection`.

## About the tuning result file
we store it in folder `data/xgb_metadata`.