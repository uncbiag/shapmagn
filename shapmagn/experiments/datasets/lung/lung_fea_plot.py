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

points_1 = source_obj['points'][:,:5000]
points_2 = target_obj['points'][:,:5000]
# fea_type_list = ["mass","dev","eigenvalue_cat","eigenvalue_prod","eigenvector_main","eigenvector"]


fea_type_list = ["mass"]
fea_extractor = feature_extractor(fea_type_list, radius=0.02, std_normalize=False, include_pos=False)
_, mass1= fea_extractor(points_1)

filter1 = mass1[..., 0] > 2.5
points_1 = points_1[filter1][None]
fea_type_list = ["eigenvalue_cat","eigenvector"]
fea_extractor = feature_extractor(fea_type_list, radius=0.025,std_normalize=False, include_pos=False)
combined_fea1, mass1 = fea_extractor(points_1)
eigenvalue, eigenvector = combined_fea1[:,:,:3], combined_fea1[:,:,3:]
eigenvector = eigenvector.view(eigenvector.shape[0], eigenvector.shape[1], 3, 3)
eigenvector_main = eigenvector[...,0]
print("detect there is {} eigenvalue smaller or equal to 0, set to 1e-7".format(torch.sum(eigenvalue<=0)))
eigenvalue[eigenvalue<=0.]=1e-7
eigenvalue = eigenvalue/torch.norm(eigenvalue,p=2, dim=2,keepdim=True)


points = points_1
points_np = points.detach().cpu().numpy().squeeze()
eigenvector_np = eigenvector.detach().cpu().numpy().squeeze()
npoints = points.shape[1]
device = points.device
sigma_scale = 0.02
principle_weight = [3,1,1]
eigenvalue[eigenvalue<0.3] = 0.3
#Gamma, principle_weight = get_Gamma(sigma_scale,principle_weight=principle_weight,eigenvalue=None,eigenvector=eigenvector)
Gamma, principle_weight = get_Gamma(sigma_scale,principle_weight=None,eigenvalue=eigenvalue,eigenvector=eigenvector)
principle_weight_np = principle_weight.squeeze().numpy()
mass,_,_ = compute_aniso_local_moments(points, Gamma)
nsample = 200
full_index = np.arange(npoints)
index = np.random.choice(full_index, nsample, replace=False)
spheres_list = [make_ellipsoid(200,radius=principle_weight_np[ind], center=points_np[ind],rotation=eigenvector_np[ind]) for ind in index]
spheres = np.concatenate(spheres_list,0)
fg_spheres_color = np.array([[0.8,.5,.2]]*len(spheres))
#visualize_point_fea(points, mass , rgb_on=False)

visualize_point_fea_with_arrow(points_1, mass1,eigenvector_main*0.01,rgb_on=False)
visualize_point_overlap(points, spheres,mass,fg_spheres_color,"aniso_filter_with_kernel_radius",point_size=[10,5],rgb_on=[False,True])






#combined_fea2, mass2 = compute_fea(points_2)
#visual_scalars1 = (visual_scalars1 - visual_scalars1.min(0)) / (visual_scalars1.max(0) - visual_scalars1.min(0))
# fea_index = 0
# visualize_point_fea_with_arrow(points_1, mass1,eigenvector_main*0.01,rgb_on=False)
# visualize_point_fea(points_2,visual_scalars2)

