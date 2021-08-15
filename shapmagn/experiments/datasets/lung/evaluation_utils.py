import os
import numpy as np
import torch
import pyvista as pv
from shapmagn.global_variable import Shape
from shapmagn.experiments.datasets.lung.global_variable import lung_expri_path
from shapmagn.datasets.vtk_utils import read_vtk
from shapmagn.shape.point_interpolator import NadWatIsoSpline
from shapmagn.utils.shape_visual_utils import save_shape_into_files
from shapmagn.utils.visualizer import (
    capture_plotter,
    visualize_source_flowed_target_overlap,
)

"""

copd10/copd10_EXP.nrrd
copd10/copd10_INSP.nrrd
copd1/copd1_EXP.nrrd
copd1/copd1_INSP.nrrd
copd2/copd2_EXP.nrrd
copd2/copd2_INSP.nrrd
copd3/copd3_EXP.nrrd
copd3/copd3_INSP.nrrd
copd4/copd4_EXP.nrrd
copd4/copd4_INSP.nrrd
copd5/copd5_EXP.nrrd
copd5/copd5_INSP.nrrd
copd6/copd6_EXP.nrrd
copd6/copd6_INSP.nrrd
copd7/copd7_EXP.nrrd
copd7/copd7_INSP.nrrd
copd8/copd8_EXP.nrrd
copd8/copd8_INSP.nrrd
copd9/copd9_EXP.nrrd
copd9/copd9_INSP.nrrd
"""


CENTER = {
    "copd1_INSP": [7.979657, 25.017563, -151.31465],
    "copd1_EXP": [8.846239, 45.1596, -142.66893],
    "copd5_INSP": [13.640025, -9.945671, -186.71013],
    "copd5_EXP": [12.206295, 8.053513, -165.32997],
    "copd8_INSP": [7.076656, 6.5697513, -167.6756],
    "copd8_EXP": [7.3253126, 13.625545, -146.99274],
    "copd4_INSP": [25.451248, 1.2760051, -136.3838],
    "copd4_EXP": [22.506023, 17.911581, -109.80095],
    "copd6_INSP": [-7.9421997e-03, -2.8869128e00, -1.4221332e02],
    "copd6_EXP": [0.782543, 12.822629, -130.49344],
    "copd9_INSP": [9.527761, 4.727795, -148.14838],
    "copd9_EXP": [13.590356, 9.209801, -135.56178],
    "copd2_INSP": [-11.987083, 13.766904, -119.20886],
    "copd2_EXP": [-13.89523, 23.859629, -122.09784],
    "copd7_INSP": [8.279412, 5.61014, -161.163],
    "copd7_EXP": [10.5092535, 10.868305, -150.65265],
    "copd3_INSP": [13.88625, 7.0715256, -174.34314],
    "copd3_EXP": [15.094385, 10.8874855, -162.57578],
    "copd10_INSP": [1.1542492, 11.651825, -163.67746],
    "copd10_EXP": [5.068997, 15.700953, -145.50748],
}

SCALE = 100


dirlab_landmarks_folder_path = os.path.join(lung_expri_path,"dirlab_landmarks")


def get_flowed(shape_pair, interp_kernel):
    flowed_points = interp_kernel(
        shape_pair.toflow.points,
        shape_pair.source.points,
        shape_pair.flowed.points,
        shape_pair.source.weights,
    )
    flowed = Shape()
    flowed.set_data_with_refer_to(flowed_points, shape_pair.toflow)
    shape_pair.set_flowed(flowed)
    return shape_pair


def get_landmarks(source_landmarks_path, target_landmarks_path):
    source_landmarks = read_vtk(source_landmarks_path)["points"]
    target_landmarks = read_vtk(target_landmarks_path)["points"]
    return source_landmarks, target_landmarks


def eval_landmark(model, shape_pair, batch_info, alias, eval_ot_map=False):
    s_name_list = batch_info["source_info"]["name"]
    t_name_list = batch_info["target_info"]["name"]
    landmarks_toflow_list, target_landmarks_list = [], []
    for s_name, t_name in zip(s_name_list, t_name_list):
        source_landmarks_path = os.path.join(
            dirlab_landmarks_folder_path, s_name + ".vtk"
        )
        target_landmarks_path = os.path.join(
            dirlab_landmarks_folder_path, t_name + ".vtk"
        )
        landmarks_toflow, target_landmarks = get_landmarks(
            source_landmarks_path, target_landmarks_path
        )
        landmarks_toflow = (landmarks_toflow - np.array(CENTER[s_name])) / SCALE
        target_landmarks = (target_landmarks - np.array(CENTER[t_name])) / SCALE
        landmarks_toflow_list.append(landmarks_toflow)
        target_landmarks_list.append(target_landmarks)
    device = shape_pair.source.points.device
    flowed_cp = shape_pair.flowed
    landmarks_toflow = torch.Tensor(np.stack(landmarks_toflow_list, 0)).to(device)
    target_landmarks_points = torch.Tensor(np.stack(target_landmarks_list, 0)).to(
        device
    )
    toflow = Shape().set_data(
        points=landmarks_toflow, weights=torch.ones_like(landmarks_toflow)
    )
    shape_pair.toflow = toflow
    gt_landmark = Shape().set_data(
        points=target_landmarks_points, weights=torch.ones_like(target_landmarks_points)
    )
    if not eval_ot_map:
        shape_pair = model.flow(shape_pair)
    else:
        interp_kernel = NadWatIsoSpline(exp_order=2, kernel_scale=0.005)
        shape_pair = get_flowed(shape_pair, interp_kernel)
    flowed_landmarks_points = shape_pair.flowed.points
    diff = (target_landmarks_points - flowed_landmarks_points) * SCALE
    shape_pair.flowed.weights, gt_landmark.weights = diff, diff
    record_path = os.path.join(
        batch_info["record_path"],
        "3d",
        "{}_epoch_{}".format(batch_info["phase"], batch_info["epoch"]),
    )
    toflow.weights = diff
    shape_pair.flowed.pointfea = None
    save_shape_into_files(
        record_path, "landmark" + alias + "_toflow", batch_info["pair_name"], toflow
    )
    save_shape_into_files(
        record_path,
        "landmark" + alias + "_flowed",
        batch_info["pair_name"],
        shape_pair.flowed,
    )
    save_shape_into_files(
        record_path,
        "landmark" + alias + "_target",
        batch_info["pair_name"],
        gt_landmark,
    )

    shape_pair.flowed = flowed_cp  # compatible to save function
    shape_pair.toflow = None  # compatible to save function

    return diff


def visualize_feature(shape_pair, batch_info):
    from sklearn.manifold import TSNE

    flowed_fea = shape_pair.flowed.pointfea.detach().cpu().numpy()
    target_fea = shape_pair.target.pointfea.detach().cpu().numpy()
    nbatch, npoints = flowed_fea.shape[0], flowed_fea.shape[1]
    camera_pos = [
        (-4.924379645467042, 2.17374925796456, 1.5003730890759344),
        (0.0, 0.0, 0.0),
        (0.40133888001174545, 0.31574165540339943, 0.8597873634998591),
    ]
    record_path = os.path.join(batch_info["record_path"], "fea_visual")
    os.makedirs(record_path, exist_ok=True)
    for b in range(nbatch):
        fea_high = np.concatenate([flowed_fea[b, :, 3:], target_fea[b, :, 3:]], 0)
        fea_embedded = TSNE(n_components=3, perplexity=30, n_jobs=5).fit_transform(
            fea_high
        )
        fea_normalized = (
            (fea_embedded - fea_embedded.min())
            / (fea_embedded.max() - fea_embedded.min() + 1e-7)
            * 3
        )
        flowed_embedding, target_embedded = (
            fea_normalized[:npoints],
            fea_normalized[npoints:],
        )
        pair_name = batch_info["pair_name"][b]
        saving_path = os.path.join(record_path, pair_name + ".png")
        visualize_source_flowed_target_overlap(
            shape_pair.source.points,
            shape_pair.flowed.points,
            shape_pair.target.points,
            flowed_embedding,
            flowed_embedding,
            target_embedded,
            title1="source",
            title2="flowed",
            title3="target",
            rgb_on=True,
            saving_capture_path=saving_path,
            camera_pos=camera_pos,
            add_bg_contrast=False,
            show=False,
        )
        data = pv.PolyData(shape_pair.source.points[b].detach().cpu().numpy())
        data.point_arrays["pointfea"] = flowed_embedding
        dcn = lambda x: x.detach().cpu().numpy()
        data.point_arrays["weights"] = dcn(shape_pair.source.weights[b])
        saving_path = os.path.join(record_path, pair_name + "_source.vtk")
        data.save(saving_path)
        data = pv.PolyData(dcn(shape_pair.flowed.points[b]))
        data.point_arrays["pointfea"] = flowed_embedding
        data.point_arrays["weights"] = dcn(shape_pair.source.weights[b])
        saving_path = os.path.join(record_path, pair_name + "_flowed.vtk")
        data.save(saving_path)
        data = pv.PolyData(dcn(shape_pair.target.points[b]))
        data.point_arrays["pointfea"] = target_embedded
        data.point_arrays["weights"] = dcn(shape_pair.target.weights[b])
        saving_path = os.path.join(record_path, pair_name + "_target.vtk")
        data.save(saving_path)


def evaluate_res(visualize_fea=False):
    def eval(metrics, shape_pair, batch_info, additional_param=None, alias=""):
        phase = batch_info["phase"]
        if phase == "val" or phase == "test":
            if visualize_fea and "mapped_position" not in additional_param:
                visualize_feature(shape_pair, batch_info)
            model = additional_param["model"]
            flowed_points_cp = shape_pair.flowed.points
            shape_pair.control_points = additional_param["initial_nonp_control_points"]
            eval_ot_map = "mapped_position" in additional_param
            has_prealign = (
                "prealign_param" in additional_param
                and additional_param["prealign_param"] is not None
            )
            record_path = os.path.join(
                batch_info["record_path"],
                "3d",
                "{}_epoch_{}".format(batch_info["phase"], batch_info["epoch"]),
            )
            os.makedirs(record_path, exist_ok=True)
            if additional_param is not None and has_prealign and not eval_ot_map:
                save_shape_into_files(
                    record_path,
                    alias + "_prealigned",
                    batch_info["pair_name"],
                    additional_param["prealigned"],
                )
                reg_param = additional_param["prealign_param"].detach().cpu().numpy()
                for pid, pair_name in enumerate(batch_info["pair_name"]):
                    np.save(
                        os.path.join(
                            record_path, pair_name + alias + "_prealigned_reg_param.npy"
                        ),
                        reg_param[pid],
                    )
            if additional_param is not None and eval_ot_map:
                shape_pair.flowed.points = additional_param["mapped_position"]
                save_shape_into_files(
                    record_path,
                    alias + "_flowed",
                    batch_info["pair_name"],
                    shape_pair.flowed,
                )
            diff = eval_landmark(
                model, shape_pair, batch_info, alias, eval_ot_map=eval_ot_map
            )
            diff_var = (diff - diff.mean(1, keepdim=True)) ** 2
            diff_var = diff_var.sum(2).mean(1)
            diff_norm_mean = diff.norm(p=2, dim=2).mean(1)
            metrics.update(
                {
                    "lmk_diff_mean"
                    + alias: [
                        _diff_norm_mean.item() for _diff_norm_mean in diff_norm_mean
                    ],
                    "lmk_diff_var"
                    + alias: [_diff_var.item() for _diff_var in diff_var],
                }
            )
            shape_pair.flowed.points = flowed_points_cp
        return metrics

    return eval
