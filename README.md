## shapmagn

shapmagn is a research project for Shape Registration. The repository provides a general framework for the point cloud/mesh registration task, supporting both optimization and learning
based approaches. Currently, we are at the early stage of the development.

## Installation
```
git clone https://github.com/uncbiag/shapmagn.git
cd shapmagn/shapmagn
pip install -r requirement.txt
cd modules/networks/pointnet2/lib
python setup.py install
```
Addtionally, torch-scatter needs to be installed, see [here](https://github.com/rusty1s/pytorch_scatter).
e.g. for cuda 10.2, 
```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
```

## Demo
For now, we provide two demos: a registration between a lung pair and a registration from sphere to cube.
```
cd shapmagn/shapmagn/demos/data
gdown https://drive.google.com/uc?id=19YG-je_7QfKd-Z8Rhg4R0nL6hVIpEco6
unzip lung_vessel_demo_data.zip
cd ..
python lung_reg.py
python toy_reg.py
```

## TODO
7. confidence map
10. test flot net, prnet
11.  maxpool
12. rewrite warp2 function in pwc
14. add transformer to geonet
17. update control point strategy (currently farthest point sampling) maybe introduce altas control points for the lung task
18. make the network more complicate to fit synthesis results
19. build an atlas distribution for the weight (radius)
20. anisotropic interpolation on spline kernel
21. test gmm model, local laplacian, main vessel