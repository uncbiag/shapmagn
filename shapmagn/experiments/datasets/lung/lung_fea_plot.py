import os, sys
sys.path.insert(0, os.path.abspath('../../../..'))
from shapmagn.datasets.data_utils import get_obj
from shapmagn.utils.shape_visual_utils import make_ellipsoid
from shapmagn.experiments.datasets.lung.lung_feature_extractor import *
# import pykeops
# pykeops.clean_pykeops()


server_path = "/home/zyshen/remote/llr11_mount/zyshen/proj/shapmagn/shapmagn/demos/" # "/playpen-raid1/"#"/home/zyshen/remote/llr11_mount/"
path_1 =  server_path+"data/lung_vessel_demo_data/10005Q_EXP_STD_NJC_COPD_wholeLungVesselParticles.vtk"
path_2 = server_path + "data/lung_vessel_demo_data/10005Q_INSP_STD_NJC_COPD_wholeLungVesselParticles.vtk"

saving_path = "/playpen-raid1/zyshen/debug/debugg_point_visual2.vtk"
reader_obj = "lung_dataset_utils.lung_reader()"
scale = -1  # an estimation of the physical diameter of the lung, set -1 for auto rescaling   #[99.90687, 65.66011, 78.61013]
normalizer_obj = "lung_dataset_utils.lung_normalizer(scale={})".format(scale)
sampler_obj = "lung_dataset_utils.lung_sampler(method='voxelgrid',scale=0.001)"
get_obj_func = get_obj(reader_obj, normalizer_obj, sampler_obj, device=torch.device("cpu"))
source_obj, source_interval = get_obj_func(path_1)
target_obj, target_interval = get_obj_func(path_2)

points_1 = source_obj['points'][:,:10000]
points_2 = target_obj['points'][:,:10000]

npoints, points = points_1.shape[1], points_1
points_np = points.detach().cpu().numpy().squeeze()
aniso_kernel_scale=0.05
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
visualize_point_overlap(points, spheres,mass,fg_spheres_color,"aniso_filter_with_kernel_radius",point_size=[10,5],rgb_on=[False,True])

