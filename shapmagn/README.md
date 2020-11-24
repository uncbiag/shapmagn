## shapmagn

shapmagn is a project for Shape Registration. The repository provides a general framework for the pointcloud/mesh registration task.


## Requirements
- High-end NVIDIA GPUs with at least 11GB of DRAM.
- Either Linux or Windows. We recommend Linux for better performance.
- CUDA Toolkit 10.1, CUDNN 7.5, and the latest NVIDIA driver.
- Python 3.6+ and PyTorch 1.4.0+.

## Installation
```
pip install -r requirement.txt
```

## Data Organization
The repository provides a general interface for shape registration task.

The data need to be organized as: task_output_folder/PHASE/data_info.json,

where PHASE include refers to "train", "val", "test" and "debug". "debug" is a subset of training data.


## Preprocess
We currently provide data preprocessing code for the lung registration. 

## Setting
All settings can be found in a "task_setting.json", where descriptions can be found at "task_setting_comment.json".