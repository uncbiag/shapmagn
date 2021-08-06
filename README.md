## Shapmagn

shapmagn is a research project for Shape Registration. The repository provides a general framework for the point cloud/mesh registration and general deep learning frameworks, supporting both optimization and learning
based approaches. 

## Installation

Please use python=3.6 to workaround a known buffer overflow bug in vtk-9.0,
Besides, to workaround background plotting issues on remote servers, we need 
```
sudo apt-get install xorg 
sudo apt-get install xvfb
pip install vtk==8.1.2

``` 

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

Here is an example on training deep feature learning network based on one synthesized pair:
```
python run_task.py -ds SHAPEMAGN_PATH/demos/data/lung_synth_dataset_splits -o SHAPEMAGN_PATH/demos/output/training_one_synth_case -tn deepfeature_pointconv -ts SHAPEMAGN_PATH/demos/settings/lung/training_deep_feature_learning_on_one_case -g 0
```
Here is an example on evaluating a pretrained deep LDDMM flow network on one real pair (the model needs to be updated):

```
python run_task.py --eval -ds SHAPEMAGN_PATH/demos/data/lung_dataset_splits -o SHAPEMAGN_PATH/demos/output/test_one_case -tn deepflow_pwc_lddmm -ts SHAPEMAGN_PATH/demos/settings/lung/test_deep_lddmm_pwcnet_on_one_case  -m   /SHAPEMAGN_PATH/demos/pretrained_models/pretrained_deep_lddmm -g 0
```

## TODO
10. test flot net, prnet
14. add transformer
21. test gmm model, local laplacian, main vessel
23. do distribution analysis for the landmarks
