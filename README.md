## Shapmagn

This is the repository for the paper "Accurate Point Cloud Registration with Robust Optimal Transport".

The repository provides a general framework for the point cloud/mesh registration and general deep learning frameworks, supporting both optimization and learning
based approaches. 


## Installation

We provide two ways to install the shapmagn: 1) custom installation  2) docker
### Custom installation
* Important. Before the installation, make sure the cuda-toolkit is installed. you can check that via "nvcc --version" in terminal. The cuda compiler
version it shows may be different from your cuda driver version shown at "nvidia-smi". Please make sure that the torch, pytorch3d and keops,torch_scatter are installed under the same cuda version as the one of nvcc.
  (Note that if your nvcc version is 11.2  while pytorch, torch_scatter of 11.2 version are not released, you can install any available version compiled with cuda 11.*)
  
* Make sure your cmake version meets the requirement [here](https://www.kernel-operations.io/keops/python/installation.html). Currently, we need cmake version >= 3.18

We assume all the following installation is under a conda virtual environment, e.g.
```
conda create -n shapmagn python=3.6
conda activate shapmagn
```
0. (Optional) if you cannot find nvcc in system, you can install via
```angular2html
conda install -c conda-forge cudatoolkit-dev=11.2
```
1. (Optional) Please use python=3.6 to workaround a known buffer overflow bug in vtk-9.0,
we recommend
   
For system support apt-get:
```
sudo apt-get install xorg 
sudo apt-get install xvfb
pip install vtk==8.1.2

```

For system support yum:
```
yum install xorg-x11-server-Xvfb
```

2. For general prediction tasks, the pytorch3d needs to be installed first [link](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md). 
   Please install all necessary packages mentioned there. 
    Essentially, pytorch3d needs pytorch to be installed first, we test using pytorch version 1.7.1. Make sure the pytorch is compiled with correct cuda version, e.g. with nvcc version=11.1, we can install pytorch
    via *conda install -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0*. However, if you have already installed cudatoolkit-dev=11.2 then don't include cudatoolkit=11.0 for pytorch installation


3. Install the Keops [link](https://www.kernel-operations.io/keops/python/installation.html)
   (make sure the current system cmake version and nvcc version meet the requirement. [Here](https://askubuntu.com/questions/829310/how-to-upgrade-cmake-in-ubuntu) is how to upgrade cmake in Ubuntu,
   After installation please run the following test to make sure Keops work)
```
import pykeops
pykeops.clean_pykeops()          # just in case old build files are still present
pykeops.test_torch_bindings()   
```

4. Install Teaser++ [link](https://teaser.readthedocs.io/en/master/installation.html)
5. Install Open3d [link](http://www.open3d.org/docs/0.7.0/getting_started.html)
   
6. Now, we can install shapmagn by
```
git clone https://github.com/uncbiag/shapmagn.git
cd shapmagn/shapmagn
pip install -r requirement.txt
cd ..
cd pointnet2/lib
python setup.py install
```
*if you use Fedora 33, you may meet a bug from a specific gcc version, you may need to downgrade gcc version via *dnf downgrade gcc*

7. torch-scatter needs to be installed, see [here](https://github.com/rusty1s/pytorch_scatter).
e.g. for cuda 11.0, 
```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
```

### Docker (doesn't support open3d and teaser++)

If you are familiar with docker, it would be much easier to run shapmagn in docker.

1. push lastest shapmagn image from dockerhub
```
docker push hbgtjxzbbx/shapmagn:v0.5
```
2. run docker locally
```
docker run --privileged --gpus all -it --rm  -v /home/zyshen/proj/shapmagn:/proj/shapmagn -v /home/zyshen/data:/data hbgtjxzbbx/shapmagn:v0.5
```
* here -v refers to the map between the local path and the docker path,
  here we map a code path and a data path based on my local env, modify local path based on your own env.

3. compile cuda code (if you use Fedora 33, you may meet a bug from a specific gcc version, you may need to downgrade gcc version via *dnf downgrade gcc*)
```
cd pointnet2/lib
python setup.py install
```

### Optional third package
For full function supports, additional package needs to be installed

1. install [probreg](https://github.com/neka-nat/probreg)
   
   (open3d version in probreg is old, some APIs have been deprecated, we recommend to install from source and fix open3d minor crashes manually)


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
python run_task.py -ds ./demos/data/lung_dataset_splits -o ./demos/output/training_feature_learning_on_one_case -tn deepfeature_pointconv_train -ts ./demos/settings/lung/deep_feature_training -g 0
```

Here is an example on robust optimal transport based deep feature projection (spline) on lung vessel dataset:
```
python run_task.py -ds ./demos/data/lung_dataset_splits -o ./demos/output/test_feature_projection_one_case -tn deepfeature_pointconv_projection -ts ./demos/settings/lung/deep_feature_projection -g 0
```

Here is an example on training a pretrained deep LDDMM flow network on one real pair:

```
python run_task.py  -ds ./demos/data/lung_dataset_splits -o ./demos/output/train_deepflow_on_one_case -tn deepflow_pwc_lddmm -ts ./demos/settings/lung/deep_lddmm_flow   -g 0
```

Here is an example on evaluating a pretrained deep LDDMM flow network on one real pair:

```
python run_task.py --eval -ds ./demos/data/lung_dataset_splits -o ./demos/output/test_deepflow_on_one_case -tn deepflow_pwc_lddmm -ts ./demos/settings/lung/deep_lddmm_flow  -m   ./demos/pretrained_models/pretrained_deep_lddmm -g 0
```