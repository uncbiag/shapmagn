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
3. add weight aug
4. real data needs unbalanced ot metric
5. synth experiment