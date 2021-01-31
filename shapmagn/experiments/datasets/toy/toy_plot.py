
import pyvista as pv
import numpy as np
import torch
from shapmagn.experiments.datasets.lung.lung_feature_extractor import feature_extractor,compute_aniso_local_moments
from shapmagn.utils.visualizer import visualize_point_fea,visualize_point_fea_with_arrow, visualize_point_overlap
from shapmagn.datasets.data_utils import compute_interval
from shapmagn.utils.shape_visual_utils import make_sphere, make_ellipsoid

###############################################################################
# Create a dataset to plot

def make_spirial_points():
    """Helper to make XYZ points"""
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 1000)
    z = np.linspace(-2, 2, 1000)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    return np.column_stack((x, y, z))


def make_line_points():
    """Helper to make XYZ points"""
    x = np.linspace(-5, 5, 1000)
    y = np.random.rand(1000)*0.001
    z = np.random.rand(1000)*0.001
    return np.column_stack((x, y, z))



def generate_aniso_gamma_on_sphere(points, foregroud_bool_index,foreground_sigma, background_sigma):
    npoints = len(points)
    nfg = sum(foregroud_bool_index)
    fg_gamma =1/(np.array(foreground_sigma)**2)
    bg_gamma =1/(np.array(background_sigma)**2)
    Gamma = np.repeat(np.diag(bg_gamma)[None],npoints,axis=0)
    fg_Gamma = np.repeat(np.diag(fg_gamma)[None],nfg,axis=0)
    Gamma[foregroud_bool_index] = fg_Gamma
    Gamma = Gamma.reshape(npoints,-1)
    return Gamma



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




def generate_eigen_vector_from_main_direction(main_direction):
    pass



# points = make_spirial_points()
# points = points.astype(np.float32)
# compute_interval(points)
# points = torch.Tensor(points)[None]
# fea_type_list= ["eignvalue_prod","eignvector_main"]
# fea_extractor = feature_extractor(fea_type_list, radius=0.1,std_normalize=False)
# combined_fea, mass = fea_extractor(points)
# pointfea, main_direction = combined_fea[..., :1], combined_fea[..., 1:]
# visualize_point_fea_with_arrow(points, mass, main_direction*0.1 , rgb_on=False)





#
# points_np = make_sphere(radius= (1.,1.,1.))
# points_np = points_np.astype(np.float32)
# npoints = len(points_np)
# bool_index = points_np[:,2].squeeze()>0.8
# compute_interval(points_np)
# fg_sigma = [0.2,0.2,0.1]
# bg_sigma = [0.05,0.05,0.05]
# Gamma = generate_aniso_gamma_on_sphere(points_np,bool_index,[0.2,0.2,0.1],[0.05,0.05,0.05])
# points = torch.Tensor(points_np)[None].contiguous()
# Gamma = torch.Tensor(Gamma)[None].contiguous()
# mass,_,_ = compute_aniso_local_moments(points,Gamma)
# fg_nsample = 3
# bg_nsample = 20
# full_index = np.arange(npoints)
# fg_index = np.random.choice(full_index[bool_index], fg_nsample, replace=False)
# bg_index = np.random.choice(full_index[np.invert(bool_index)], bg_nsample, replace=False)
# fg_spheres_list = [make_sphere(600,radius=fg_sigma, center=points_np[ind]) for ind in fg_index]
# bg_spheres_list = [make_sphere(50,radius=bg_sigma, center=points_np[ind]) for ind in bg_index]
# fg_spheres = np.concatenate(fg_spheres_list,0)
# bg_spheres = np.concatenate(bg_spheres_list,0)
# fg_spheres_color = np.array([[0.8,.5,.2]]*len(fg_spheres))
# bg_spheres_color = np.array([[-1.2,-1,-1.2]]*len(bg_spheres))
# surrounding_spheres = np.concatenate([fg_spheres,bg_spheres],0)
# surrounding_spheres_color = np.concatenate([fg_spheres_color,bg_spheres_color],0)
# #visualize_point_fea(points, mass , rgb_on=False)
# visualize_point_overlap(points, surrounding_spheres,mass,surrounding_spheres_color,"aniso_filter_with_kernel_radius",point_size=[10,5],rgb_on=[False,True])




points = make_spirial_points()
points = points.astype(np.float32)
points = torch.tensor(points)[None]
fea_type_list = ["eignvalue_cat","eignvector"]
fea_extractor = feature_extractor(fea_type_list, radius=0.1,std_normalize=False, include_pos=False)
combined_fea1, mass1 = fea_extractor(points)
eignvalue, eignvector = combined_fea1[:,:,:3], combined_fea1[:,:,3:]
eignvector = eignvector.view(eignvector.shape[0], eignvector.shape[1], 3, 3)
eignvector_main = eignvector[...,0]
# eignvalue is not used, but compute to check abnormal
print("detect there is {} eigenvalue smaller or equal to 0, set to 1e-7".format(torch.sum(eignvalue<=0)))
eignvalue[eignvalue<=0.]=1e-7

points_np = points.detach().cpu().numpy().squeeze()
eignvector_np = eignvector.detach().cpu().numpy().squeeze()
npoints = points.shape[1]
device = points.device
sigma_scale = 0.2
principle_vector = [3,1,1]
Gamma, principle_vector = get_Gamma(sigma_scale,principle_vector,eignvector)
mass,_,_ = compute_aniso_local_moments(points, Gamma)
nsample = 30
full_index = np.arange(npoints)
index = np.random.choice(full_index, nsample, replace=False)
spheres_list = [make_ellipsoid(200,radius=principle_vector, center=points_np[ind],rotation=eignvector_np[ind]) for ind in index]
spheres = np.concatenate(spheres_list,0)
fg_spheres_color = np.array([[0.8,.5,.2]]*len(spheres))
#visualize_point_fea(points, mass , rgb_on=False)
visualize_point_fea_with_arrow(points, mass1,eignvector_main*0.05,rgb_on=False)
visualize_point_overlap(points, spheres,mass,fg_spheres_color,"aniso_filter_with_kernel_radius",point_size=[10,5],rgb_on=[False,True])





