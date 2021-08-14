import os, sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../.."))
from shapmagn.utils.module_parameters import ParameterDict
from shapmagn.datasets.data_utils import get_pair_obj
from shapmagn.shape.shape_pair_utils import create_shape_pair
from shapmagn.models_reg.multiscale_optimization import (
    build_single_scale_model_embedded_solver,
)
from shapmagn.global_variable import MODEL_POOL, Shape, shape_type
from shapmagn.utils.visualizer import visualize_source_flowed_target_overlap
from shapmagn.demos.demo_utils import *
from shapmagn.experiments.datasets.toy.toy_utils import *

# import pykeops
# pykeops.clean_pykeops()

assert (
    shape_type == "pointcloud"
), "set shape_type = 'pointcloud'  in global_variable.py"
device = torch.device("cpu")  # cuda:0  cpu
reader_obj = "flyingkitti_nonocc_utils.flyingkitti_nonocc_reader(flying3d=False)"
normalizer_obj = "flyingkitti_nonocc_utils.flyingkitti_nonocc_normalizer()"
sampler_obj = "flyingkitti_nonocc_utils.flyingkitti_nonocc_sampler(num_sample=20000)"
pair_postprocess_obj = (
    "flyingkitti_nonocc_utils.flyingkitti_nonocc_pair_postprocess(flying3d=False)"
)
pair_postprocess = obj_factory(pair_postprocess_obj)

assert (
    shape_type == "pointcloud"
), "set shape_type = 'pointcloud'  in global_variable.py"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
server_path = "./"  # "/playpen-raid1/"#"/home/zyshen/remote/llr11_mount/"
source_path = server_path + "data/kitti_data/000000/pc1.npy"
target_path = server_path + "data/kitti_data/000000/pc2.npy"
get_obj_func = get_pair_obj(
    reader_obj,
    normalizer_obj,
    sampler_obj,
    pair_postprocess_obj,
    device,
    expand_bch_dim=True,
)
source_obj, target_obj, source_interval, target_interval = get_obj_func(
    source_path, target_path
)
min_interval = min(source_interval, target_interval)
print(
    "the source and the target min interval is {},{}".format(
        source_interval, target_interval
    )
)
input_data = {"source": source_obj, "target": target_obj}
create_shape_pair_from_data_dict = obj_factory(
    "shape_pair_utils.create_source_and_target_shape()"
)
source, target = create_shape_pair_from_data_dict(input_data)
shape_pair = create_shape_pair(source, target)

##############  do registration ###########################s############

""" Experiment 1:  gradient flow """
task_name = "gradient_flow"
solver_opt = ParameterDict()
record_path = server_path + "output/flyingkitti_reg/{}".format(task_name)
os.makedirs(record_path, exist_ok=True)
solver_opt["record_path"] = record_path
model_name = "gradient_flow_opt"
model_opt = ParameterDict()
model_opt[
    "interpolator_obj"
] = "point_interpolator.nadwat_kernel_interpolator(scale=0.1, exp_order=2)"
model_opt[("sim_loss", {}, "settings for sim_loss_opt")]
model_opt["sim_loss"]["loss_list"] = ["geomloss"]
model_opt["sim_loss"][("geomloss", {}, "settings for geomloss")]
model_opt["sim_loss"]["geomloss"]["attr"] = "points"
blur = 0.1
model_opt["sim_loss"]["geomloss"][
    "geom_obj"
] = "geomloss.SamplesLoss(loss='sinkhorn',blur={}, scaling=0.8,debias=False)".format(
    blur
)
model = MODEL_POOL[model_name](model_opt)
solver = build_single_scale_model_embedded_solver(solver_opt, model)
model.init_reg_param(shape_pair)
shape_pair = solver(shape_pair)
print("the registration complete")
gif_folder = os.path.join(record_path, "gif")
os.makedirs(gif_folder, exist_ok=True)
saving_gif_path = os.path.join(gif_folder, task_name + ".gif")
fea_to_map = shape_pair.source.points[0]
mapped_fea = get_omt_mapping(
    model_opt["sim_loss"]["geomloss"],
    source,
    target,
    fea_to_map,
    p=2,
    mode="hard",
    confid=0.0,
)
flow_points = shape_pair.flowed.points - shape_pair.source.points
visualize_source_flowed_target_overlap(
    shape_pair.source.points,
    shape_pair.flowed.points,
    shape_pair.target.points,
    fea_to_map,
    fea_to_map,
    mapped_fea,
    "source",
    "gradient_flow",
    "target",
    rgb_on=[True, True, True],
    saving_gif_path=None,
)
