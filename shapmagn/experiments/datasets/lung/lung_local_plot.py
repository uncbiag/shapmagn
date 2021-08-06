import os, sys
import numpy as np

from shapmagn.experiments.datasets.lung.lung_data_analysis import get_half_lung

sys.path.insert(0, os.path.abspath('../../../..'))
from shapmagn.datasets.data_utils import get_obj
from shapmagn.global_variable import *
from shapmagn.utils.shape_visual_utils import make_ellipsoid
from shapmagn.experiments.datasets.lung.lung_feature_extractor import compute_anisotropic_gamma_from_points
from shapmagn.utils.visualizer import visualize_point_fea_with_arrow,visualize_point_overlap
# import pykeops
# pykeops.clean_pykeops()


server_path = "/home/zyshen/remote/llr11_mount/zyshen/proj/shapmagn/shapmagn/demos/" # "/playpen-raid1/"#"/home/zyshen/remote/llr11_mount/"
path_1 =  server_path+"data/lung_vessel_demo_data/case1_exp.vtk"
path_2 = server_path + "data/lung_vessel_demo_data/case1_insp.vtk"

saving_path = "/playpen-raid1/zyshen/debug/debugg_point_visual2.vtk"
reader_obj = "lung_dataloader_utils.lung_reader()"
scale = -1  # an estimation of the physical diameter of the lung, set -1 for auto rescaling   #[99.90687, 65.66011, 78.61013]
normalizer_obj = "lung_dataloader_utils.lung_normalizer(scale={})".format([100,100,100])
sampler_obj = "lung_dataloader_utils.lung_sampler( method='combined',scale=0.0003,num_sample=60000,sampled_by_weight=True)"
get_obj_func = get_obj(reader_obj, normalizer_obj, sampler_obj, device=torch.device("cpu"))
source_obj, source_interval = get_obj_func(path_1)
target_obj, target_interval = get_obj_func(path_2)

points_1 = source_obj['points']
weights_1 = source_obj["weights"]
points_2 = target_obj['points']

npoints, points = points_1.shape[1], points_1
points_np = points.detach().cpu().numpy().squeeze()
aniso_kernel_scale=0.02
Gamma,principle_weight,eigenvector, mass = compute_anisotropic_gamma_from_points(points,cov_sigma_scale=0.02,aniso_kernel_scale=aniso_kernel_scale,leaf_decay=True,principle_weight=None,eigenvalue_min=0.1,iter_twice=True,return_details=True)
principle_weight_np = principle_weight.squeeze().numpy()
eigenvector_np = eigenvector.squeeze().numpy()
nsample = 700
full_index = np.arange(npoints)
index = np.random.choice(full_index, nsample, replace=False)
spheres_list = [make_ellipsoid(200,radius=principle_weight_np[ind], center=points_np[ind],rotation=eigenvector_np[ind]) for ind in index]
spheres = np.concatenate(spheres_list,0)
fg_spheres_color = np.array([[0.8,.5,.2]]*len(spheres))
#visualize_point_fea(points, mass , rgb_on=False)

visualize_point_fea_with_arrow(points_1, mass,eigenvector[:,:,:,0]*0.01,rgb_on=False)
visualize_point_overlap(points, spheres,weights_1,fg_spheres_color,"aniso_filter_with_kernel_radius",point_size=[15,8],rgb_on=[False,True])

