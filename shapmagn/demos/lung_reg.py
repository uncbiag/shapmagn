import os, sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))
import numpy as np
import torch
from shapmagn.utils.module_parameters import ParameterDict
from shapmagn.utils.obj_factory import obj_factory
from shapmagn.datasets.data_utils import get_file_name, generate_pair_name, get_obj
from shapmagn.shape.shape_pair_utils import create_shape_pair
from shapmagn.models.multiscale_optimization import build_single_scale_model_embedded_solver, build_multi_scale_solver
from shapmagn.global_variable import MODEL_POOL,Shape, shape_type
from shapmagn.utils.utils import get_grid_wrap_points
from shapmagn.utils.visualizer import *
from shapmagn.demos.demo_utils import *
from shapmagn.experiments.datasets.lung.lung_data_analysis import *
# import pykeops
# pykeops.clean_pykeops()


def analysis(shape_pair, fea_to_map, mapped_fea, compute_on_half_lung=True, saving_path=None):
    source = shape_pair.source
    target = shape_pair.target
    flowed = shape_pair.flowed

    visualize_multi_point(points_list=[source.points,flowed.points,target.points],
                         feas_list=[fea_to_map,fea_to_map, mapped_fea],
                         titles_list=["source", "gradient_flow","target"],
                         rgb_on=[True, True, True],
                         saving_gif_path=None if not saving_path else os.path.join(saving_path,"s_f_t_full.gif"),
                          saving_capture_path=None if not saving_path else os.path.join(saving_path,"s_f_t_full.png"))

    # #
    # #if the computation has already based on half computation, here we don't need to refilter again
    source_half = get_half_lung(source) if not compute_on_half_lung else source
    target_half = get_half_lung(target) if not compute_on_half_lung else target
    flowed_half = get_half_lung(flowed) if not compute_on_half_lung else flowed
    visualize_multi_point(points_list=[source_half.points, flowed_half.points, target_half.points],
                          feas_list=[source_weight_transform(source_half.weights, compute_on_half_lung),
                                     flowed_weight_transform(flowed_half.weights, compute_on_half_lung),
                                     target_weight_transform(target_half.weights, compute_on_half_lung)],
                          titles_list=["source", "gradient_flow", "target"],
                          rgb_on=[False, False, False],
                          saving_gif_path=None if not saving_path else os.path.join(saving_path, "s_f_t_main.gif"),
                          saving_capture_path=None if not saving_path else os.path.join(saving_path, "s_f_t_main.png"))

    visualize_point_pair_overlap(source_half.points, target_half.points,
                             source_weight_transform(source_half.weights,compute_on_half_lung),
                             target_weight_transform(target_half.weights,compute_on_half_lung),
                             title1="source",title2="target", rgb_on=False,
                                 saving_gif_path=None if not saving_path else os.path.join(saving_path,
                                                                                           "s_t_overlap.gif"),
                                 saving_capture_path=None if not saving_path else os.path.join(saving_path,
                                                                                               "s_t_overlap.png"))
    visualize_point_pair_overlap(flowed_half.points, target_half.points,
                             flowed_weight_transform(flowed_half.weights,compute_on_half_lung),
                             target_weight_transform(target_half.weights,compute_on_half_lung),
                             title1="flowed",title2="target", rgb_on=False,
                                 saving_gif_path=None if not saving_path else os.path.join(saving_path,
                                                                                           "ft_overlap.gif"),
                                 saving_capture_path=None if not saving_path else os.path.join(saving_path,
                                                                                               "f_t_overlap.png"))

# set shape_type = "pointcloud"  in global_variable.py
assert shape_type == "pointcloud", "set shape_type = 'pointcloud'  in global_variable.py"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
server_path = "./" # "/playpen-raid1/"#"/home/zyshen/remote/llr11_mount/"
source_path =  server_path+"data/lung_vessel_demo_data/10031R_EXP_STD_NJC_COPD_wholeLungVesselParticles.vtk"
target_path = server_path + "data/lung_vessel_demo_data/10031R_INSP_STD_NJC_COPD_wholeLungVesselParticles.vtk"
compute_on_half_lung = True

####################  prepare data ###########################
pair_name = generate_pair_name([source_path,target_path])
reader_obj = "lung_dataset_utils.lung_reader()"
scale = -1 # an estimation of the physical diameter of the lung, set -1 for auto rescaling   #[99.90687, 65.66011, 78.61013]
normalizer_obj = "lung_dataset_utils.lung_normalizer(scale={})".format(scale)
sampler_obj = "lung_dataset_utils.lung_sampler(method='voxelgrid',scale=0.001)"
get_obj_func = get_obj(reader_obj,normalizer_obj,sampler_obj, device)
source_obj, source_interval = get_obj_func(source_path)
target_obj, target_interval = get_obj_func(target_path)
min_interval = min(source_interval,target_interval)
input_data = {"source":source_obj,"target":target_obj}
source_target_generator = obj_factory("shape_pair_utils.create_source_and_target_shape()")
source, target = source_target_generator(input_data)
source = get_half_lung(source,normalize_weight=True) if compute_on_half_lung else source
target = get_half_lung(target,normalize_weight=True) if compute_on_half_lung else target
# source = get_key_vessel(source,thre=2e-5)
# target = get_key_vessel(target,thre=1.4e-5)
shape_pair = create_shape_pair(source, target)
#
################  do registration ###########################s############

""" Experiment 1:  gradient flow """
task_name = "gradient_flow"
solver_opt = ParameterDict()
record_path = server_path+"output/lung_demo/{}".format(task_name)
os.makedirs(record_path,exist_ok=True)
solver_opt["record_path"] = record_path
solver_opt["save_2d_capture_every_n_iter"] = 1
solver_opt["capture_plot_obj"] = "lung_data_analysis.capture_plotter()"
model_name = "gradient_flow_opt"
model_opt =ParameterDict()
model_opt["interpolator_obj"] ="point_interpolator.kernel_interpolator(scale=0.01, exp_order=2)"
model_opt[("sim_loss", {}, "settings for sim_loss_opt")]
model_opt['sim_loss']['loss_list'] =  ["geomloss"]
model_opt['sim_loss'][("geomloss", {}, "settings for geomloss")]
model_opt['sim_loss']['geomloss']["attr"] = "pointfea"

# the feature extractor should be disabled in gradient flow.  we leave it here to show to observe the feature behavior.
# if not explicitly set, the points position would be taken as the pointfea
# model_opt["pair_feature_extractor_obj"] ="lung_feature_extractor.pair_feature_extractor(fea_type_list=['eignvalue_prod'],weight_list=[0.1], radius=0.05,include_pos=True)"

blur = 0.005
model_opt['sim_loss']['geomloss']["geom_obj"] = "geomloss.SamplesLoss(loss='sinkhorn',blur={}, scaling=0.8,reach=1,debias=False)".format(blur)
model = MODEL_POOL[model_name](model_opt)
solver = build_single_scale_model_embedded_solver(solver_opt,model)
model.init_reg_param(shape_pair)
shape_pair=solver(shape_pair)
print("the registration complete")
gif_folder = os.path.join(record_path,"gif")
os.makedirs(gif_folder,exist_ok=True)
saving_gif_path = os.path.join(gif_folder,task_name+".gif")
fea_to_map =  shape_pair.source.points[0]
shape_pair.source, shape_pair.target = model.extract_fea(shape_pair.source, shape_pair.target)
mapped_fea = get_omt_mapping(model_opt['sim_loss']['geomloss'], shape_pair.source, shape_pair.target,fea_to_map , blur= blur,p=2,mode="hard",confid=0.0)
analysis(shape_pair, fea_to_map, mapped_fea, compute_on_half_lung=True)


# """ Experiment 2: lddmm flow  too slow !!!, and likely to experience numerical underflow, see expri 3 for a workaround"""
# task_name = "lddmm"
# solver_opt = ParameterDict()
# record_path = server_path+"output/lung_demo/{}".format(task_name)
# os.makedirs(record_path,exist_ok=True)
# solver_opt["record_path"] = record_path
# solver_opt["point_grid_scales"] =  [0.08, -1]
# solver_opt["iter_per_scale"] = [50, 200]
# solver_opt["rel_ftol_per_scale"] = [ 1e-9,1e-9,]
# solver_opt["init_lr_per_scale"] = [5e-1,5e-2]
# solver_opt["save_3d_shape_every_n_iter"] = 20
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
#
# """ Experiment 3: lddmm guide by gradient flow """
# task_name = "gradient_flow_guided_by_lddmm"
# solver_opt = ParameterDict()
# record_path = server_path+"output/lung_demo/{}".format(task_name)
# os.makedirs(record_path,exist_ok=True)
# solver_opt["record_path"] = record_path
# solver_opt["save_2d_capture_every_n_iter"] = 5
# solver_opt["capture_plot_obj"] = "lung_data_analysis.capture_plotter()"
# solver_opt["point_grid_scales"] =  [ -1]
# solver_opt["iter_per_scale"] = [60]
# solver_opt["rel_ftol_per_scale"] = [ 1e-9]
# solver_opt["init_lr_per_scale"] = [5e-1]
# solver_opt["save_3d_shape_every_n_iter"] = 5
# solver_opt["shape_sampler_type"] = "point_grid"
# solver_opt["stragtegy"] = "use_optimizer_defined_here"
# solver_opt[("optim", {}, "setting for the optimizer")]
# solver_opt[("scheduler", {}, "setting for the scheduler")]
# solver_opt["optim"]["type"] = "sgd" #lbgfs
# solver_opt["scheduler"]["type"] = "step_lr"
# solver_opt["scheduler"][("step_lr",{},"settings for step_lr")]
# solver_opt["scheduler"]["step_lr"]["gamma"] = 0.5
# solver_opt["scheduler"]["step_lr"]["step_size"] = 30
#
# model_name = "lddmm_opt"
# model_opt =ParameterDict()
# module_name = "hamiltonian" #hamiltonian variational
# model_opt["module"] =module_name
# model_opt[(module_name, {}, "settings for lddmm")]
# model_opt[module_name]['kernel'] =  "keops_kernels.LazyKeopsKernel(kernel_type='multi_gauss', sigma_list=[0.03,0.05, 0.08],weight_list=[0.2,0.3, 0.5])"
# model_opt["use_gradflow_guided"] = True
# model_opt[("gradflow_guided", {}, "settings for gradflow guidance")]
# model_opt["gradflow_guided"] ['update_gradflow_every_n_step']= 10
# model_opt["gradflow_guided"] ['gradflow_blur_init']= 0.05
# model_opt["gradflow_guided"] ['update_gradflow_blur_by_raito']= 0.5
# model_opt["gradflow_guided"] ['gradflow_blur_min']= 0.05
# model_opt[("sim_loss", {}, "settings for sim_loss_opt")]
# model_opt['sim_loss']['loss_list'] =  ["l2"]
# model_opt['sim_loss']['l2']["attr"] = "points"
# model_opt['sim_loss'][("geomloss", {}, "settings for geomloss")]
# model_opt['sim_loss']['geomloss']["attr"] = "points"
# model_opt['sim_loss']['geomloss']["geom_obj"] = "geomloss.SamplesLoss(loss='sinkhorn',blur=placeholder, scaling=0.8,debias=False)"
#
#
# model = MODEL_POOL[model_name](model_opt)
# model.init_reg_param(shape_pair)
# solver = build_multi_scale_solver(solver_opt,model)
# solver(shape_pair)
# print("the registration complete")
# gif_folder = os.path.join(record_path,"gif")
# os.makedirs(gif_folder,exist_ok=True)
# saving_gif_path = os.path.join(gif_folder,task_name+".gif")
# fea_to_map =  shape_pair.source.points[0]
# blur = 0.001
# mapped_fea = get_omt_mapping(model_opt['sim_loss']['geomloss'], source, target,fea_to_map , blur= blur,p=2,mode="hard",confid=0.1)
# visualize_multi_point([shape_pair.source.points[0],shape_pair.flowed.points[0],shape_pair.target.points[0]],
#                      [fea_to_map,fea_to_map, mapped_fea],
#                      ["source", "gradient_flow","target"],
#                         [True, True, True],
#                       saving_path=None)
#
# #
# #
#
# experiment 4: discrete flow with graident flow as initializer

# part1  gradient flow prealign

# task_name = "gradient_flow"
# solver_opt = ParameterDict()
# record_path = server_path+"output/lung_demo/{}".format(task_name)
# os.makedirs(record_path,exist_ok=True)
# solver_opt["record_path"] = record_path
# model_name = "gradient_flow_opt"
# model_opt =ParameterDict()
# model_opt["interpolator_obj"] ="point_interpolator.kernel_interpolator(scale=0.01, exp_order=2)"
# model_opt[("sim_loss", {}, "settings for sim_loss_opt")]
# model_opt['sim_loss']['loss_list'] =  ["geomloss"]
# model_opt['sim_loss'][("geomloss", {}, "settings for geomloss")]
# model_opt['sim_loss']['geomloss']["attr"] = "pointfea"
# blur = 0.08
# model_opt['sim_loss']['geomloss']["geom_obj"] = "geomloss.SamplesLoss(loss='sinkhorn',blur={}, scaling=0.8,reach=1,debias=False)".format(blur)
# model = MODEL_POOL[model_name](model_opt)
# solver = build_single_scale_model_embedded_solver(solver_opt,model)
# model.init_reg_param(shape_pair)
# shape_pair=solver(shape_pair)



#
#
# # part2 discrete flow
# task_name = "discrete_flow"
# solver_opt = ParameterDict()
# record_path = server_path+"output/lung_demo/{}".format(task_name)
# solver_opt["record_path"] = record_path
# solver_opt["save_2d_capture_every_n_iter"] = 10
# solver_opt["capture_plot_obj"] = "lung_data_analysis.capture_plotter()"
# solver_opt["point_grid_scales"] =  [-1]
# solver_opt["iter_per_scale"] = [200]
# solver_opt["rel_ftol_per_scale"] = [ 1e-9,1e-9,1e-9]
# solver_opt["init_lr_per_scale"] = [1e0,5e-2,1e-4]
# solver_opt["save_3d_shape_every_n_iter"] = 20
# solver_opt["shape_sampler_type"] = "point_grid"
# solver_opt["stragtegy"] = "use_optimizer_defined_here"
# solver_opt[("optim", {}, "setting for the optimizer")]
# solver_opt[("scheduler", {}, "setting for the scheduler")]
# solver_opt["optim"]["type"] = "sgd" #lbgfs
# solver_opt["scheduler"]["type"] = "step_lr"
# solver_opt["scheduler"][("step_lr",{},"settings for step_lr")]
# solver_opt["scheduler"]["step_lr"]["gamma"] = 0.5
# solver_opt["scheduler"]["step_lr"]["step_size"] = 80
# model_name = "discrete_flow_opt"
# model_opt =ParameterDict()
# model_opt["drift_every_n_iter"] = 20
# model_opt["interpolator_obj"] ="point_interpolator.kernel_interpolator(scale=0.01, exp_order=2)"
# model_opt["apply_spline_kernel"]= True
# model_opt["gauss_smoother_sigma"] = 0.02
# #model_opt["pair_feature_extractor_obj"] ="lung_feature_extractor.pair_feature_extractor(fea_type_list=['eignvalue_prod'],weight_list=[0.1], radius=0.05,include_pos=True)"
# model_opt[("sim_loss", {}, "settings for sim_loss_opt")]
# model_opt['sim_loss']['loss_list'] = ["geomloss"]
# model_opt['sim_loss'][("geomloss", {}, "settings for geomloss")]
# model_opt['sim_loss']['geomloss']["attr"] = "pointfea"
# blur = 0.001
# model_opt['sim_loss']['geomloss']["geom_obj"] = "geomloss.SamplesLoss(loss='sinkhorn',blur={}, scaling=0.8, debias=True)".format(blur)
# model = MODEL_POOL[model_name](model_opt)
# solver = build_multi_scale_solver(solver_opt,model)
# model.init_reg_param(shape_pair)
# shape_pair = solver(shape_pair)
# print("the registration complete")
#

#
# # experiment 5: feature mapping
# blur = 0.0005
# pair_feature_extractor_obj = "lung_feature_extractor.pair_feature_extractor(fea_type_list=['eignvalue_prod'],weight_list=[0.1], radius=0.01,include_pos=True)"
# pair_feature_extractor = obj_factory(pair_feature_extractor_obj)
# geomloss_opt = ParameterDict()
# geomloss_opt["attr"] = "pointfea"
# geomloss_opt["geom_obj"] = "geomloss.SamplesLoss(loss='sinkhorn',blur={}, scaling=0.8, debias=True)".format(blur)
# source, target = pair_feature_extractor(shape_pair.source,shape_pair.target)
# fea_to_map = shape_pair.source.points[0]
# mapped_fea = get_omt_mapping(geomloss_opt, source, target,fea_to_map , blur= blur,p=2,mode="hard",confid=0.0)
# source_mask=(shape_pair.source.weights>2e-05).squeeze()
# target_mask=(shape_pair.target.weights>1.4e-05).squeeze()
# visualize_point_pair(shape_pair.source.points[0][source_mask],shape_pair.target.points[0][target_mask],
#                      fea_to_map[source_mask],mapped_fea[target_mask],
#                      "source","target",
#                       saving_path=None)




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

