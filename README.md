## shapmagn

shapmagn is a research project for Shape Registration. The repository provides a general framework for the point cloud/mesh registration task, supporting both optimization and learning
based approaches. Currently, we are at the early stage of the development.

## Installation
```
git clone https://github.com/uncbiag/shapmagn.git
cd shapmagn/shapmagn
pip install -r requirement.txt
```
Addtionally, torch-scatter needs to be installed, see [here](https://github.com/rusty1s/pytorch_scatter).
e.g. for cuda 10.2, 
```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
```

## Demo
```
cd shapmagn/shapmagn/demos/data
gdown https://drive.google.com/uc?id=19YG-je_7QfKd-Z8Rhg4R0nL6hVIpEco6
unzip lung_vessel_demo_data.zip
cd ..
python gradient_lung_reg.py
```