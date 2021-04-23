# Prepare the data
```
cd shapmagn/experiments/datasets/lung
```
## run lung_prepare_data.py 
you need to set four paths here.  the "data_folder_path" and "dirlab_folder_path" refer to the 1000 lung dataset and DirLab COPD dataset.

```
data_folder_path = "PATH/UNC_vesselParticles"
dirlab_folder_path = "PATH/DIRLABVascular"
data_output_folder_path = "PATH/OUTPUT_DATASET_FOLDER"
dirlab_val_output_folder = "PATH/OUTPUT_DIRLAB_FOLDER"
```
By default, four folders will be created in "data_output_folder_path", which refers to the "train", "val", "test", "debug*" splits. 

*Here the debug split refers to a subset of training set.

In our experiments, we use the "OUTPUT_DIRLAB_FOLDER" to replace the "val" split

## (optional), visualization and analysis
1. lung_local_plot.py visualizes the anistropic kernel on vessels
2. lung_data_anlsysis.py computes and matches the lung radius distribution


# Optimization tasks

The optimization-based lung demos can be found at shapmagn/demos/lung_reg.py
It includes several representative experiments include: prealign, gradient flow, coherent point drift (spline-based), LDDMM, etc.


# Learning tasks

The learning-based taks can simply run by

```
python run_task.py -ds=PATH_TO_OUTPUT_DATASET_FOLDER -o OUTPUT_FOLDER_PATH -tn TASK_NAME -ts PATH_TO_TASK_SETTING_FOLDER -g GPU_IDS(can be mulitiple gpus)
```

Here is an example on training deep feature learning network based on one case:
```
python run_task.py -ds SHAPEMAGN_PATH/shapmagn/demos/data/lung_dataset_splits -o SHAPEMAGN_PATH/shapmagn/demos/output/training_one_case -tn deepfeature_pointnet2 -ts SHAPEMAGN_PATH/shapmagn/demos/settings/lung/training_deep_feature_learning_on_one_case -g 0```
```
Here is an example on evaluate a pretrained deep LLDDMM flow network on one case:

```
python run_task.py --eval -ds SHAPEMAGN_PATH/shapmagn/demos/data/lung_dataset_splits -o SHAPEMAGN_PATH/shapmagn/demos/output/test_one_case -tn deepflow_pwc_lddmm -ts SHAPEMAGN_PATH/shapmagn/demos/settings/lung/test_deep_lddmm_pwcnet_on_one_case  -m   /SHAPEMAGN_PATH/shapmagn/demos/pretrained_models/pretrained_deep_lddmm -g 0```

```
