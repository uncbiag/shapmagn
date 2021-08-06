import os, sys
import torch
from shapmagn.utils.visualizer import (
    visualize_point_pair,
    visualize_point_pair_overlap,
    visualize_source_flowed_target_overlap,
    visualize_point_overlap,
)
from shapmagn.global_variable import Shape
from shapmagn.datasets.vtk_utils import read_vtk


def init_shpae(points_path):
    if points_path:
        shape = Shape()
        shape_dict = read_vtk(points_path)
        points, weights = torch.Tensor(shape_dict["points"][None]), torch.Tensor(
            shape_dict["weights"][None]
        )
        shape.set_data(points=points, weights=weights)
        return shape
    else:
        return None


folder_path = "/home/zyshen/remote/llr11_mount/zyshen/data/flying3d_nonocc_test_on_kitti/model_eval/draw/deepflow_spline_8192_withaug_kitti_prealigned/records/3d/test_epoch_-1"
folder_path = "/home/zyshen/remote/llr11_mount/zyshen/data/flying3d_nonocc_larger_kitti/model_eval/draw/deepflow_spline_8192_withaug_kitti_prealigned/records/3d/test_epoch_-1"
# folder_path ="/home/zyshen/remote/llr11_mount/zyshen/data/lung_expri/model_eval/deep_feature_pointconv_dirlab_complex_aniso_15dim_normalized_60000_nonsmooth/records/3d/test_epoch_-1"
case_id = "000003"
prealigned_output_path = os.path.join(folder_path, case_id + "_prealigned.png")
nonp_output_path = os.path.join(folder_path, case_id + "_nonp.png")
gf_output_path = os.path.join(folder_path, case_id + "_nonp_gf.png")
source_path = os.path.join(folder_path, case_id + "_source.vtk")
target_path = os.path.join(folder_path, case_id + "_target.vtk")
landmark_path = os.path.join(folder_path, case_id + "_landmark_gf_target.vtk")
prealigned_path = os.path.join(folder_path, case_id + "__prealigned.vtk")
reg_param_path = case_id
nonp_path = os.path.join(folder_path, case_id + "_flowed.vtk")
nonp_gf_path = os.path.join(folder_path, case_id + "__gf_flowed.vtk")
source = init_shpae(source_path)
target = init_shpae(target_path)
prealigned = init_shpae(prealigned_path)
nonp = init_shpae(nonp_path)
nonp_gf = init_shpae(nonp_gf_path)
# landmark = init_shpae(landmark_path)
camera_pos = [
    (12.24069705556629, 154.18144864069825, 52.5302855418096),
    (0.0, 0.0, 0.0),
    (0.17653893360765158, 0.08828369494012701, -0.9803264732365397),
]
visualize_source_flowed_target_overlap(
    source.points,
    nonp.points,
    target.points,
    source.points,
    nonp.points,
    target.points,
    "source",
    "nonp_gf",
    "target",
    rgb_on=False,
    add_bg_contrast=True,
    camera_pos=camera_pos,
    saving_capture_path=None,
    show=True,
)


# visualize_point_overlap(target.points, landmark.points, target.weights, landmark.weights,"landmark_target",point_size=(5,10),opacity=(0.05, 1),
#                                        rgb_on=False, camera_pos=camera_pos,saving_capture_path=prealigned_output_path,show=True)
#

# visualize_point_pair_overlap(source.points,target.points,  nonp.weights, target.weights,"prealigned","target",
#                                        rgb_on=False, camera_pos=camera_pos,saving_capture_path=prealigned_output_path,show=True)

# visualize_point_pair_overlap(prealigned.points,target.points,  prealigned.weights, target.weights,"prealigned","target",
#                                        rgb_on=False, camera_pos=camera_pos,saving_capture_path=prealigned_output_path,show=True)
#
# visualize_point_pair_overlap(nonp.points,target.points,  nonp.weights, target.weights,"nonp","target",
#                                        rgb_on=False, camera_pos=camera_pos,saving_capture_path=prealigned_output_path,show=True)
#
# visualize_point_pair_overlap(nonp_gf.points,target.points,  nonp_gf.weights, target.weights,"nonp_gf","target",
#                                        rgb_on=False, camera_pos=camera_pos,saving_capture_path=prealigned_output_path,show=True)
