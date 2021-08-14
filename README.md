## Shapmagn

This is the repository for the paper "Accurate Point Cloud Registration with Robust Optimal Transport".

The repository provides a general framework for the point cloud/mesh registration and general deep learning frameworks, supporting both optimization and learning
based approaches. 


## Installation

Please use python=3.6 to workaround a known buffer overflow bug in vtk-9.0,
To saving plots at remote servers, we suggest to install xvfb
```
sudo apt-get install xorg 
sudo apt-get install xvfb
pip install vtk==8.1.2

``` 

For shape registration tasks, the pytorch3d is not needed. But for general prediction tasks, e.g, landmark prediction. Make sure install [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md) first.

Finally, we can install shapmagn by
```
git clone https://github.com/uncbiag/shapmagn.git
cd shapmagn/shapmagn
pip install -r requirement.txt
cd ..
cd pointnet2/lib
python setup.py install
```
torch-scatter needs to be installed, see [here](https://github.com/rusty1s/pytorch_scatter).
e.g. for cuda 10.2, 
```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
```

Make sure the torch use the same cuda version as keops and torch-scatter.

## Demo
We provide a series of demos, which can be found at shapmagn/demos
Here are two examples on how to run the optimization-based demos :

```
cd shapmagn/shapmagn/demos/data
gdown https://drive.google.com/uc?id=19YG-je_7QfKd-Z8Rhg4R0nL6hVIpEco6
unzip lung_vessel_demo_data.zip
cd ..
python 2d_toy_reg.py
python toy_reg.py
python partial_prealign_reg.py
python lung_reg.py
python flyingkitti_reg.py
```

Here is an example on deep feature learning on lung vessel dataset:
```
python run_task.py -ds ./demos/data/lung_synth_dataset_splits -o ./demos/output/training_one_synth_case -tn deepfeature_pointconv_train -ts ./demos/settings/lung/deep_feature_training -g 0
```

Here is an example on robust optimal transport based deep feature projection (spline) on lung vessel dataset:
```
python run_task.py -ds ./demos/data/lung_synth_dataset_splits -o ./demos/output/training_one_synth_case -tn deepfeature_pointconv_projection -ts ./demos/settings/lung/deep_feature_projection -g 0
```

Here is an example on training a pretrained deep LDDMM flow network on one real pair:

```
python run_task.py  -ds ./demos/data/lung_dataset_splits -o ./demos/output/test_one_case -tn deepflow_pwc_lddmm -ts ./demos/settings/lung/deep_lddmm_flow   -g 0
```

Here is an example on evaluating a pretrained deep LDDMM flow network on one real pair:

```
python run_task.py --eval -ds ./demos/data/lung_dataset_splits -o ./demos/output/test_one_case -tn deepflow_pwc_lddmm -ts ./demos/settings/lung/deep_lddmm_flow  -m   ./demos/pretrained_models/pretrained_deep_lddmm -g 0
```