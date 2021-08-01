import os, sys
import torch
from shapmagn.utils.visualizer import visualize_point_pair, visualize_point_pair_overlap, \
    visualize_source_flowed_target_overlap, visualize_point_overlap
from shapmagn.global_variable import Shape
from shapmagn.datasets.vtk_utils import read_vtk


ID_COPD={
"12042G":"copd6",
"12105E":"copd7",
"12109M":"copd8",
"12239Z":"copd9",
"12829U":"copd10",
"13216S":"copd1",
"13528L":"copd2",
"13671Q":"copd3",
"13998W":"copd4",
"17441T":"copd5"
}


def init_shpae( points_path):
    if points_path:
        shape = Shape()
        shape_dict = read_vtk(points_path)
        points, weights = torch.Tensor(shape_dict["points"][None]), torch.Tensor(shape_dict["weights"][None])
        shape.set_data(points=points, weights = weights)
        return shape
    else:
        return None

folder_path ="/home/zyshen/remote/llr11_mount/zyshen/data/lung_expri/model_eval/draw/deep_flow_prealign_pwc_lddmm_4096_new_60000_8192_aniso_rerun/records/3d/test_epoch_-1"
# folder_path ="/home/zyshen/remote/llr11_mount/zyshen/data/lung_expri/model_eval/draw/deep_flow_prealign_pwc2_2_continue_60000/records/3d/test_epoch_-1"
# folder_path ="/home/zyshen/remote/llr11_mount/zyshen/data/lung_expri/model_eval/draw/deep_flow_prealign_pwc_spline_4096_new_60000_8192_aniso_rerun_debug/records/3d/test_epoch_-1"
#folder_path ="/home/zyshen/remote/llr11_mount/zyshen/data/lung_expri/model_eval/deep_feature_pointconv_dirlab_complex_aniso_15dim_normalized_60000/records/3d/test_epoch_-1"
#folder_path ="/home/zyshen/remote/llr11_mount/zyshen/data/lung_expri/model_eval/deep_feature_pointconv_dirlab_complex_aniso_15dim_normalized_60000_nonsmooth/records/3d/test_epoch_-1"
case_id = "13998W"
prealigned_output_path = os.path.join(folder_path,case_id+"_prealigned.png")
nonp_output_path = os.path.join(folder_path,case_id+"_nonp.png")
gf_output_path = os.path.join(folder_path,case_id+"_nonp_gf.png")
source_path = os.path.join(folder_path, case_id+"_source.vtk")
target_path =  os.path.join(folder_path, case_id+"_target.vtk")
landmark_path = os.path.join(folder_path,case_id+"_landmark_gf_target.vtk")
prealigned_path = os.path.join(folder_path,  case_id + "__prealigned.vtk")
reg_param_path = case_id
nonp_path =  os.path.join(folder_path, case_id +"_flowed.vtk")
nonp_gf_path =  os.path.join(folder_path, case_id+ "__gf_flowed.vtk")
source = init_shpae(source_path)
target = init_shpae(target_path)
prealigned = init_shpae(prealigned_path)
nonp = init_shpae(nonp_path)
nonp_gf = init_shpae(nonp_gf_path)
landmark = init_shpae(landmark_path)
camera_pos=[(-4.924379645467042, 2.17374925796456, 1.5003730890759344),(0.0, 0.0, 0.0),(0.40133888001174545, 0.31574165540339943, 0.8597873634998591)]
visualize_source_flowed_target_overlap(source.points,nonp_gf.points, target.points, source.weights, nonp_gf.weights, target.weights, "source","nonp_gf","target",
                                       rgb_on=False, add_bg_contrast=False,camera_pos=camera_pos,saving_capture_path=None,show=True)


visualize_point_overlap(target.points, landmark.points, target.weights, landmark.weights,"landmark_target",point_size=(5,10),opacity=(0.05, 1),
                                       rgb_on=False, camera_pos=camera_pos,saving_capture_path=prealigned_output_path,show=True)


visualize_point_pair_overlap(source.points,target.points,  source.weights, target.weights,"source","target",
                                       rgb_on=False, camera_pos=camera_pos,saving_capture_path=None,show=True)

visualize_point_pair_overlap(prealigned.points,target.points,  prealigned.weights, target.weights,"prealigned","target",
                                       rgb_on=False, camera_pos=camera_pos,saving_capture_path=prealigned_output_path,show=True)

visualize_point_pair_overlap(nonp.points,target.points,  nonp.weights, target.weights,"nonparametric","target",
                                       rgb_on=False, camera_pos=camera_pos,saving_capture_path=prealigned_output_path,show=True)

visualize_point_pair_overlap(nonp_gf.points,target.points,  nonp_gf.weights, target.weights,"nonparametric_then_fintuned","target",
                                       rgb_on=False, camera_pos=camera_pos,saving_capture_path=prealigned_output_path,show=True)
