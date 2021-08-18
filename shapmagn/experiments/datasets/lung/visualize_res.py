from os.path import join
import torch

# The main "visual" routines:
from shapmagn.utils.visualizer import (
    visualize_point_pair,
    visualize_point_pair_overlap,
    visualize_source_flowed_target_overlap,
    visualize_point_overlap,
)
from shapmagn.global_variable import Shape
from shapmagn.datasets.vtk_utils import read_vtk


# Conversion table between our IDs and the
# standard IDs from the DirLab-COPD dataset:
ID_COPD = {
    "copd6": "copd6",
    "copd7": "copd7",
    "copd8": "copd8",
    "copd9": "copd9",
    "copd10": "copd10",
    "copd1": "copd1",
    "copd2": "copd2",
    "copd3": "copd3",
    "copd4": "copd4",
    "copd5": "copd5",
}
ID_DATA = {v: k for k, v in ID_COPD.items()}


def init_shape(points_path):
    """Turns a .vtk filename into a shape object with .points and .weights attributes."""
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


# Which folder are we going to read?
folder_root = "/home/zyshen/remote/llr11_mount/zyshen/data/lung_expri/model_eval/"

folder_suffix = "/records/3d/test_epoch_-1"

# Some experiments:
experiment_name = "draw/deep_flow_prealign_pwc_lddmm_4096_new_60000_8192_aniso_rerun"
experiment_name = "deep_flow_prealign_pwc2_2_continue_60000"
experiment_name = (
    "draw/deep_flow_prealign_pwc_spline_4096_new_60000_8192_aniso_rerun_debug"
)
experiment_name = "deep_feature_pointconv_dirlab_complex_aniso_15dim_normalized_60000"
experiment_name = (
    "deep_feature_pointconv_dirlab_complex_aniso_15dim_normalized_60000_nonsmooth"
)

# experiment_name = "deepflow/disp"
experiment_name = "deepflow/lddmm"
# experiment_name = "deepflow/spline"
# experiment_name = "deepfeature/opt_discrete_flow_deep"

folder_path = folder_root + experiment_name + folder_suffix

# Our subject:
copd_id = 2  # Any number in [1, 10] is fine
case_id = ID_DATA[f"copd{copd_id}"]


# Detailed paths for the PNG screenshots (output):


def filename(s):
    return join(folder_path, case_id + s)


# Shapes from the VTK files (input):
def get_shape(s):
    return init_shape(filename(s))


source = get_shape("_source.vtk")
target = get_shape("_target.vtk")
prealigned = get_shape("__prealigned.vtk")
nonp = get_shape("_flowed.vtk")
finetuned = get_shape("__gf_flowed.vtk")

landmarks_source = get_shape("_landmark_gf_toflow.vtk")
landmarks_nonp = get_shape("_landmark_flowed.vtk")
landmarks_finetuned = get_shape("_landmark_gf_flowed.vtk")
landmarks_target = get_shape("_landmark_gf_target.vtk")

# Camera position:
camera_pos = [
    (-4.924379645467042, 2.17374925796456, 1.5003730890759344),
    (0.0, 0.0, 0.0),
    (0.40133888001174545, 0.31574165540339943, 0.8597873634998591),
]

constant_kwargs = {
    "show": False,
    "rgb_on": False,
    "camera_pos": camera_pos,
}

visualize_source_flowed_target_overlap(
    source.points,
    finetuned.points,
    target.points,
    source.weights,
    finetuned.weights,
    target.weights,
    "source",
    "finetuned",
    "target",
    saving_capture_path=filename(f"_1_overview.jpg"),
    add_bg_contrast=False,
    **constant_kwargs,
)


snapshots = [
    (source, "source", "_2_source.jpg"),
    (prealigned, "prealigned", "_3_prealigned.jpg"),
    (nonp, "nonparametric", "_4_nonparametric.jpg"),
    (finetuned, "finetuned", "_5_finetuned.jpg"),
]

for shape, name, suffix in snapshots:
    visualize_point_pair_overlap(
        shape.points,
        target.points,
        shape.weights,
        target.weights,
        name,
        "target",
        saving_capture_path=filename(suffix),
        **constant_kwargs,
    )


snapshots = [
    (source, landmarks_source, "source", "source", "_6_source.jpg"),
    (nonp, landmarks_nonp, "nonparametric", "source", "_7_nonparametric.jpg"),
    (finetuned, landmarks_finetuned, "finetuned", "source", "_8_finetuned.jpg"),
    (target, landmarks_target, "target", "target", "_9_target.jpg"),
]

for shape, landmarks, name, color, suffix in snapshots:

    visualize_point_overlap(
        shape.points,
        landmarks.points,
        shape.weights,
        landmarks.weights,
        name,
        color=color,
        opacity=(0.15, 1),
        saving_capture_path=filename(suffix),
        **constant_kwargs,
    )
