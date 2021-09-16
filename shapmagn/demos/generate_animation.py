import os
import torch
import pyvista as pv
from shapmagn.utils.generate_animation import init_shpae, FlowModel, camera_pos_interp, generate_gif, visualize_animation
from shapmagn.shape.shape_pair_utils import create_shape_pair
from shapmagn.utils.module_parameters import ParameterDict
def reader():
    def read(file_info):
        path = file_info["data_path"]
        data = pv.read(path)
        data_dict = {}
        data_dict["points"] = data.points.astype(np.float32)
        data_dict["faces"] = data.faces.reshape(-1,4)[:,1:].astype(np.int32)
        for name in data.array_names:
            try:
                data_dict[name] = data[name]
            except:
                pass
        return data_dict
    return read

import numpy as np
case_id = "glass_head"
folder_path = "./output/gif"
source_path = os.path.join(folder_path, case_id+"_source.vtk")
target_path =  os.path.join(folder_path, case_id+"_target.vtk")
rigid_param_path = os.path.join(folder_path, "rigid.npy")
prealign_reg_param = torch.Tensor(np.load(rigid_param_path))[None]

total_captrue_path_list = []
source = init_shpae(source_path, reader)
target = init_shpae(target_path, reader)
output_folder = os.path.join(folder_path, "gif", case_id, "prealign")
os.makedirs(output_folder, exist_ok=True)
stage_name = "rigid"
model_type = "affine_interp"
flow_opt = ParameterDict()
flow_opt["model_type"] = model_type
flow_opt["t_list"] = list(np.linspace(0, 1.0, num=20))
target_list = [target] * len(flow_opt["t_list"])
flow_model = FlowModel(flow_opt)
shape_pair = create_shape_pair(source, target, pair_name=case_id, n_control_points=-1)
shape_pair.reg_param = prealign_reg_param
flowed_list = flow_model(shape_pair)
camera_pos_start = [(3.512278346100715, 4.250464101608384, 11.322542330791322),
 (5.1975250244140625e-05, -0.01716797798871994, 0.030868902802467346),
 (-0.0535002383947687, 0.939470467204134, -0.3384271941540001)]
camera_pos_end = [(9.162664509092975, 9.516713946663296, 7.557944172548784),
 (0.004772096872329712, -0.007005035877227783, 0.016719788312911987),
 (-0.46940519278623677, 0.7796239004251746, -0.41454232458357415)]
pos_interp_list = camera_pos_interp(camera_pos_start, camera_pos_end, flow_opt["t_list"])
saving_capture_path_list = [os.path.join(output_folder, "t_{:.2f}.png").format(t) for t in flow_opt["t_list"]]
title1_list = [stage_name] * len(flow_opt["t_list"])
title2_list = ["target"] * len(flow_opt["t_list"])
for _flowed in flowed_list:
    _flowed.weights = source.points

for _target in target_list:
    _target.weights = target.points

visualize_animation(flowed_list,target_list,title1_list,title2_list,rgb_on=False,saving_capture_path_list=saving_capture_path_list,camera_pos_list=pos_interp_list,show=False)
total_captrue_path_list += saving_capture_path_list
total_captrue_path_list += [total_captrue_path_list[-1]] * 10

gif_path = os.path.join(folder_path,"gif",case_id,"reg5.gif")
generate_gif(total_captrue_path_list,gif_path)