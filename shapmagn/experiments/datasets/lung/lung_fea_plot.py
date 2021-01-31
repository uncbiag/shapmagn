import os, sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))
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
# fea_type_list = ["mass","dev","eignvalue_cat","eignvalue_prod","eignvector_main","eignvector"]


fea_type_list = ["mass"]
fea_extractor = feature_extractor(fea_type_list, radius=0.02, std_normalize=False, include_pos=False)
_, mass1= fea_extractor(points_1)

filter1 = mass1[..., 0] > 2.5
points_1 = points_1[filter1][None]
fea_type_list = ["eignvalue_cat","eignvector"]
fea_extractor = feature_extractor(fea_type_list, radius=0.025,std_normalize=False, include_pos=False)
combined_fea1, mass1 = fea_extractor(points_1)
eignvalue, eignvector = combined_fea1[:,:,:3], combined_fea1[:,:,3:]
eignvector = eignvector.view(eignvector.shape[0], eignvector.shape[1], 3, 3)
eignvector_main = eignvector[...,0]
print("detect there is {} eigenvalue smaller or equal to 0, set to 1e-7".format(torch.sum(eignvalue<=0)))
eignvalue[eignvalue<=0.]=1e-7
eigenvalue = eignvalue/torch.norm(eignvalue,p=2, dim=2,keepdim=True)

def get_Gamma(sigma_scale, principle_vector=[1,1,1], eignvector=None):
    """

    :param sigma_scale: scale the sigma
    :param principle_vector: a list of sigma of D size
    :param eignvector: 1XNxDxD
    :return:
    """
    device = eignvector.device
    nbatch, npoints = eignvector.shape[0], eignvector.shape[1]
    assert nbatch == 1
    principle_vector = np.array(principle_vector)/np.linalg.norm(principle_vector)*sigma_scale
    principle_vector = principle_vector.astype(np.float32)
    principle_vector_inv_sq = 1/(principle_vector**2)
    principle_diag = torch.diag(torch.tensor(principle_vector_inv_sq).to(device)).repeat(1,npoints,1,1) # 1xNxDxD
    Gamma = eignvector @ principle_diag @ (eignvector.permute(0,1,3,2))
    return Gamma.view(nbatch,npoints,-1),principle_vector

points = points_1
points_np = points.detach().cpu().numpy().squeeze()
eignvector_np = eignvector.detach().cpu().numpy().squeeze()
npoints = points.shape[1]
device = points.device
sigma_scale = 0.02
principle_vector = [3,1,1]
Gamma, principle_vector = get_Gamma(sigma_scale,principle_vector,eignvector)
mass,_,_ = compute_aniso_local_moments(points, Gamma)
nsample = 200
full_index = np.arange(npoints)
index = np.random.choice(full_index, nsample, replace=False)
spheres_list = [make_ellipsoid(200,radius=principle_vector, center=points_np[ind],rotation=eignvector_np[ind]) for ind in index]
spheres = np.concatenate(spheres_list,0)
fg_spheres_color = np.array([[0.8,.5,.2]]*len(spheres))
#visualize_point_fea(points, mass , rgb_on=False)

visualize_point_fea_with_arrow(points_1, mass1,eignvector_main*0.01,rgb_on=False)
visualize_point_overlap(points, spheres,mass,fg_spheres_color,"aniso_filter_with_kernel_radius",point_size=[10,5],rgb_on=[False,True])






#combined_fea2, mass2 = compute_fea(points_2)
#visual_scalars1 = (visual_scalars1 - visual_scalars1.min(0)) / (visual_scalars1.max(0) - visual_scalars1.min(0))
# fea_index = 0
# visualize_point_fea_with_arrow(points_1, mass1,eignvector_main*0.01,rgb_on=False)
# visualize_point_fea(points_2,visual_scalars2)

