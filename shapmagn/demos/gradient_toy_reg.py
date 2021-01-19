import os, sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))
import numpy as np
import torch
from shapmagn.utils.module_parameters import ParameterDict
from shapmagn.utils.obj_factory import obj_factory
from shapmagn.datasets.data_utils import compute_interval
from shapmagn.datasets.data_utils import get_file_name, generate_pair_name
from shapmagn.shape.shape_pair_utils import create_shape_pair
from shapmagn.models.multiscale_optimization import build_single_scale_model_embedded_solver, build_multi_scale_solver
from shapmagn.global_variable import MODEL_POOL,Shape, shape_type
from shapmagn.utils.utils import get_grid_wrap_points
from shapmagn.utils.visualizer import visualize_point_fea, visualize_point_pair, visualize_multi_point
# import pykeops
# pykeops.clean_pykeops()

# set shape_type = "pointcloud"  in global_variable.py
assert shape_type == "pointcloud", "set shape_type = 'pointcloud'  in global_variable.py"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
server_path = "./" # "/playpen-raid1/"#"/home/zyshen/remote/llr11_mount/"
source_path =  server_path+"data/toy_demo_data/divide_3d_sphere_level1.vtk"
target_path = server_path + "data/toy_demo_data/divide_3d_cube_level4.vtk"



def get_obj(reader_obj,normalizer_obj,sampler_obj):
    def _get_obj(file_path):
        name = get_file_name(file_path)
        file_info = {"name":name,"data_path":file_path}
        reader = obj_factory(reader_obj)
        normalizer = obj_factory(normalizer_obj)
        sampler = obj_factory(sampler_obj)
        raw_data_dict  = reader(file_info)
        normalized_data_dict = normalizer(raw_data_dict)
        sampled_data_dict = sampler(normalized_data_dict)
        min_interval = compute_interval(sampled_data_dict["points"])
        obj_dict = sampled_data_dict
        obj = {key: torch.from_numpy(fea)[None].to(device) for key, fea in obj_dict.items()}
        return obj, min_interval
    return _get_obj



def detect_folding(warped_grid_points, grid_size,spacing, saving_path=None,file_name=None):
    from shapmagn.utils.img_visual_utils import compute_jacobi_map
    from shapmagn.utils.utils import point_to_grid
    warped_grid = point_to_grid(warped_grid_points,grid_size)
    compute_jacobi_map(warped_grid[None],spacing,saving_path,[file_name])


def get_omt_mapping(gemloss_setting, source, target, fea_to_map, blur=0.01, p=2,mode="hard", confid=0.1):
    # here we assume batch_sz = 1
    from shapmagn.metrics.losses import GeomDistance
    from pykeops.torch import LazyTensor
    geom_obj = gemloss_setting["geom_obj"].replace(")",",potentials=True)")
    geomloss = obj_factory(geom_obj)
    attr = gemloss_setting['attr']
    attr1 = getattr(source, attr)
    attr2 = getattr(target, attr)
    weight1 = source.weights[:, :, 0]  # remove the last dim
    weight2 = target.weights[:, :, 0]  # remove the last dim
    F_i, G_j  = geomloss(weight1, attr1, weight2, attr2) # todo batch sz of input and output in geomloss is not consistent

    N,M,D = source.points.shape[1], target.points.shape[1],  source.points.shape[2]
    a_i, x_i = LazyTensor(source.weights.view(N,1,1)), LazyTensor(source.points.view(N, 1, D))
    b_j, y_j = LazyTensor(target.weights.view(1, M,1)), LazyTensor(target.points.view(1, M, D))
    F_i, G_j = LazyTensor(F_i.view(N, 1,1)), LazyTensor(G_j.view(1, M,1))
    C_ij = (1 / p) * ((x_i - y_j) ** p).sum(-1)  # (N,M,1) cost matrix
    eps = blur ** p  # temperature epsilon
    P_j = ((F_i + G_j - C_ij) / eps).exp() * (a_i)  # (N,M,1) transport plan
    if mode=="soft":
        fea_to_map = LazyTensor(fea_to_map.view(N, 1, -1))  # Nx1xfea_dim
        mapped_fea = (P_j*fea_to_map).sum_reduction(0) #(N,M,fea_dim)-> (M,fea_dim)
    elif mode == "hard":
        P_j_max, P_j_index = P_j.max_argmax(0)
        mapped_fea = fea_to_map[P_j_index][:,0]
        below_thre_index = (P_j_max<confid)[:,0]
        mapped_fea[below_thre_index] = 0
    elif mode == "confid":
        # P_j_max, P_j_index = P_j.max_argmax(0)
        # mapped_fea = P_j_max
        mapped_fea = P_j.sum_reduction(0)
    else:
        raise ValueError("mode {} not defined, support: soft/ hard/ confid".format(mode))
    return mapped_fea


####################  prepare data ###########################
pair_name = generate_pair_name([source_path,target_path])
reader_obj = "toy_dataset_utils.toy_reader()"
sampler_obj = "toy_dataset_utils.toy_sampler()"
normalizer_obj = "toy_dataset_utils.toy_normalizer()"
get_obj_func = get_obj(reader_obj,normalizer_obj,sampler_obj)
source_obj, source_interval = get_obj_func(source_path)
target_obj, target_interval = get_obj_func(target_path)
min_interval = min(source_interval,target_interval)
input_data = {"source":source_obj,"target":target_obj}
source_target_generator = obj_factory("shape_pair_utils.create_source_and_target_shape()")
source, target = source_target_generator(input_data)
shape_pair = create_shape_pair(source, target)

#################  do registration ###########################s############

""" Experiment 1:  gradient flow """
task_name = "gradient_flow"
solver_opt = ParameterDict()
record_path = server_path+"output/toy_demo/{}".format(task_name)
os.makedirs(record_path,exist_ok=True)
solver_opt["record_path"] = record_path
model_name = "gradient_flow_opt"
model_opt =ParameterDict()
model_opt["interpolator_obj"] ="point_interpolator.kernel_interpolator(scale=0.1, exp_order=2)"
model_opt[("sim_loss", {}, "settings for sim_loss_opt")]
model_opt['sim_loss']['loss_list'] =  ["geomloss"]
model_opt['sim_loss'][("geomloss", {}, "settings for geomloss")]
model_opt['sim_loss']['geomloss']["attr"] = "points"
blur = 0.005
model_opt['sim_loss']['geomloss']["geom_obj"] = "geomloss.SamplesLoss(loss='sinkhorn',blur={}, scaling=0.8, reach=0.1,debias=False)".format(blur)
model = MODEL_POOL[model_name](model_opt)
solver = build_single_scale_model_embedded_solver(solver_opt,model)
model.init_reg_param(shape_pair)
solver(shape_pair)
print("the registration complete")
gif_folder = os.path.join(record_path,"gif")
os.makedirs(gif_folder,exist_ok=True)
saving_gif_path = os.path.join(gif_folder,task_name+".gif")
fea_to_map =  shape_pair.source.points[0]
mapped_fea = get_omt_mapping(model_opt['sim_loss']['geomloss'], source, target,fea_to_map , blur= blur,p=2,mode="hard",confid=0.1)
visualize_multi_point([shape_pair.source.points[0],shape_pair.flowed.points[0],shape_pair.target.points[0]],
                     [fea_to_map,fea_to_map, mapped_fea],
                     ["source", "gradient_flow","target"],
                        [True, True, True],
                      saving_path=None)


#
# """ Experiment 2: lddmm flow  too slow !!! """
# task_name = "lddmm"
# solver_opt = ParameterDict()
# record_path = server_path+"output/toy_demo/{}".format(task_name)
# os.makedirs(record_path,exist_ok=True)
# solver_opt["record_path"] = record_path
# solver_opt["point_grid_scales"] =  [0.08, -1]
# solver_opt["iter_per_scale"] = [50, 200]
# solver_opt["rel_ftol_per_scale"] = [ 1e-9,1e-9,]
# solver_opt["init_lr_per_scale"] = [5e-1,5e-2]
# solver_opt["save_every_n_iter"] = 20
# solver_opt["shape_sampler_type"] = "point_grid"
# solver_opt["stragtegy"] = "use_optimizer_defined_here"
# solver_opt[("optim", {}, "setting for the optimizer")]
# solver_opt[("scheduler", {}, "setting for the scheduler")]
# solver_opt["optim"]["type"] = "sgd" #lbgfs
# solver_opt["scheduler"]["type"] = "step_lr"
# solver_opt["scheduler"][("step_lr",{},"settings for step_lr")]
# solver_opt["scheduler"]["step_lr"]["gamma"] = 0.5
# solver_opt["scheduler"]["step_lr"]["step_size"] = 80
# model_name = "lddmm_opt"
# model_opt =ParameterDict()
# model_opt["module"] ="hamiltonian"
# model_opt[("hamiltonian", {}, "settings for hamiltonian")]
# model_opt['hamiltonian']['kernel'] =  "keops_kernels.LazyKeopsKernel(kernel_type='multi_gauss', sigma_list=[0.008,0.02],weight_list=[0.5, 0.5])"
# model_opt[("sim_loss", {}, "settings for sim_loss_opt")]
# model_opt['sim_loss']['loss_list'] =  ["geomloss"]
# model_opt['sim_loss'][("geomloss", {}, "settings for geomloss")]
# model_opt['sim_loss']['geomloss']["attr"] = "points"
# model_opt['sim_loss']['geomloss']["geom_obj"] = "geomloss.SamplesLoss(loss='sinkhorn',blur=0.0005, scaling=0.8, debias=True)"
# model = MODEL_POOL[model_name](model_opt)
# solver = build_multi_scale_solver(solver_opt,model)
# model.init_reg_param(shape_pair)
# solver(shape_pair)
# print("the registration complete")



#
# """ Experiment 3: lddmm guide by gradient flow """
# task_name = "gradient_flow_guided_by_lddmm"
# solver_opt = ParameterDict()
# record_path = server_path+"output/toy_demo/{}".format(task_name)
# os.makedirs(record_path,exist_ok=True)
# solver_opt["record_path"] = record_path
# solver_opt["point_grid_scales"] =  [ -1]
# solver_opt["iter_per_scale"] = [50]
# solver_opt["rel_ftol_per_scale"] = [ 1e-9,]
# solver_opt["init_lr_per_scale"] = [5e-1]
# solver_opt["save_every_n_iter"] = 10
# solver_opt["shape_sampler_type"] = "point_grid"
# solver_opt["stragtegy"] = "use_optimizer_defined_here"
# solver_opt[("optim", {}, "setting for the optimizer")]
# solver_opt[("scheduler", {}, "setting for the scheduler")]
# solver_opt["optim"]["type"] = "sgd" #lbgfs
# solver_opt["scheduler"]["type"] = "step_lr"
# solver_opt["scheduler"][("step_lr",{},"settings for step_lr")]
# solver_opt["scheduler"]["step_lr"]["gamma"] = 0.5
# solver_opt["scheduler"]["step_lr"]["step_size"] = 80
#
# model_name = "lddmm_opt"
# model_opt =ParameterDict()
# model_opt["module"] ="hamiltonian"
# model_opt[("hamiltonian", {}, "settings for hamiltonian")]
# model_opt['hamiltonian']['kernel'] =  "keops_kernels.LazyKeopsKernel(kernel_type='multi_gauss', sigma_list=[0.01,0.03, 0.05],weight_list=[0.2,0.3, 0.5])"
# model_opt["use_gradflow_guided"] = True
# model_opt[("gradflow_guided", {}, "settings for gradflow guidance")]
# model_opt["gradflow_guided"] ['gradflow_guided_every_step']= 10
# model_opt[("sim_loss", {}, "settings for sim_loss_opt")]
# model_opt['sim_loss']['loss_list'] =  ["l2"]
# model_opt['sim_loss']['l2']["attr"] = "points"
# model_opt['sim_loss'][("geomloss", {}, "settings for geomloss")]
# model_opt['sim_loss']['geomloss']["attr"] = "points"
# blur = 0.005
# model_opt['sim_loss']['geomloss']["geom_obj"] = "geomloss.SamplesLoss(loss='sinkhorn',blur={}, scaling=0.8, debias=True)".format(blur)
#
#
# model = MODEL_POOL[model_name](model_opt)
# model.init_reg_param(shape_pair)
# solver = build_multi_scale_solver(solver_opt,model)
# solver(shape_pair)
# print("the registration complete")





######################### folding detections ##########################################
source_grid_spacing = np.array([0.05]*3).astype(np.float32) #max(source_interval*20, 0.01)
source_wrap_grid, grid_size = get_grid_wrap_points(source_obj["points"][0], source_grid_spacing)
source_wrap_grid = source_wrap_grid[None]
toflow = Shape()
toflow.set_data(points=source_wrap_grid)
shape_pair.set_toflow(toflow)
shape_pair.control_weights = torch.ones_like(shape_pair.control_weights)/shape_pair.control_weights.shape[1]
model.flow(shape_pair)
detect_folding(shape_pair.flowed.points,grid_size,source_grid_spacing,record_path,pair_name)

