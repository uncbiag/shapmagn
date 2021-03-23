"""keops failed on multi-processing data loader,  so the synthsis process are put before the network"""
import os, sys
sys.path.insert(0, os.path.abspath('../../../..'))
import random
import time
from shapmagn.experiments.datasets.lung.lung_data_analysis import *
from shapmagn.global_variable import *
from shapmagn.datasets.data_aug import SplineAug, PointAug
from shapmagn.utils.module_parameters import ParameterDict
from shapmagn.utils.utils import enlarge_by_factor
from functools import partial



def lung_synth_data(**kwargs):
    aug_settings = ParameterDict()
    aug_settings["do_local_deform_aug"] = kwargs["do_rigid_aug"] if "do_rigid_aug" in kwargs else True
    aug_settings["do_grid_aug"] = kwargs["do_rigid_aug"] if "do_rigid_aug" in kwargs else True
    aug_settings["do_point_aug"] = kwargs["do_rigid_aug"] if "do_rigid_aug" in kwargs else True
    aug_settings["do_rigid_aug"] = kwargs["do_rigid_aug"] if "do_rigid_aug" in kwargs else False
    aug_settings["plot"] = False
    local_deform_aug = aug_settings[
        ("local_deform_aug", {}, "settings for uniform sampling based spline augmentation")]
    local_deform_aug["num_sample"] = 1000
    local_deform_aug["disp_scale"] = 0.05
    kernel_scale = 0.12
    spline_param = "cov_sigma_scale=0.02,aniso_kernel_scale={},eigenvalue_min=0.3,iter_twice=True, fixed=False, leaf_decay=False, is_interp=True".format(
        kernel_scale)
    local_deform_aug['local_deform_spline_kernel_obj'] = "point_interpolator.NadWatAnisoSpline(exp_order=2,{})".format(
        spline_param)
    grid_spline_aug = aug_settings[("grid_spline_aug", {}, "settings for grid sampling based spline augmentation")]
    grid_spline_aug["grid_spacing"] = 0.9
    grid_spline_aug["disp_scale"] = 0.12
    kernel_scale = 0.1
    grid_spline_aug[
        "grid_spline_kernel_obj"] = "point_interpolator.NadWatIsoSpline(kernel_scale={}, exp_order=2)".format(
        kernel_scale)
    rigid_aug_settings = aug_settings[
        ("rigid_aug", {}, "settings for rigid augmentation")]
    rigid_aug_settings["rotation_range"] = [-50,50]
    rigid_aug_settings["scale_range"] = [0.8, 1.2]
    rigid_aug_settings["translation_range"] = [-0.3,0.3]


    spline_aug = SplineAug(aug_settings)



    points_aug = aug_settings[
        ("points_aug", {}, "settings for remove or add noise points")]
    points_aug["remove_random_points"] = False
    points_aug["add_random_point_noise"] = False
    points_aug["add_random_weight_noise"] = True
    points_aug["remove_random_points_by_ratio"] = 0.01
    points_aug["add_random_point_noise_by_ratio"] = 0.01
    points_aug["random_weight_noise_scale"] = 0.1
    points_aug["random_noise_raidus"] = 0.1
    points_aug["normalize_weights"] = False
    points_aug["plot"] = False
    point_aug = PointAug(points_aug)

    def _synth(data_dict):
        synth_info = {"aug_settings":aug_settings}
        points, weights = data_dict["points"],data_dict["weights"]
        if aug_settings["do_point_aug"]:
            points, weights, corr_index = point_aug(points, weights)
            synth_info["corr_index"] = corr_index

        if aug_settings["do_local_deform_aug"] or aug_settings["do_spline_aug"]:
            points, weights = spline_aug(points, weights)
        data_dict["points"], data_dict["weights"] = points, weights
        return data_dict, synth_info
    return _synth



def lung_aug_data(**kwargs):
    aug_settings = ParameterDict()
    aug_settings["do_local_deform_aug"] = kwargs["do_rigid_aug"] if "do_rigid_aug" in kwargs else True
    aug_settings["do_grid_aug"] = kwargs["do_rigid_aug"] if "do_rigid_aug" in kwargs else True
    aug_settings["do_point_aug"] = kwargs["do_rigid_aug"] if "do_rigid_aug" in kwargs else True
    aug_settings["do_rigid_aug"] = kwargs["do_rigid_aug"] if "do_rigid_aug" in kwargs else False
    aug_settings["plot"] = False
    local_deform_aug = aug_settings[
        ("local_deform_aug", {}, "settings for uniform sampling based spline augmentation")]
    local_deform_aug["num_sample"] = 1000
    local_deform_aug["disp_scale"] = 0.05
    kernel_scale = 0.12
    spline_param = "cov_sigma_scale=0.02,aniso_kernel_scale={},eigenvalue_min=0.3,iter_twice=True, fixed=False, leaf_decay=False, is_interp=True".format(
        kernel_scale)
    local_deform_aug['local_deform_spline_kernel_obj'] = "point_interpolator.NadWatAnisoSpline(exp_order=2,{})".format(
        spline_param)
    grid_spline_aug = aug_settings[("grid_spline_aug", {}, "settings for grid sampling based spline augmentation")]
    grid_spline_aug["grid_spacing"] = 0.9
    grid_spline_aug["disp_scale"] = 0.05
    kernel_scale = 0.1
    grid_spline_aug[
        "grid_spline_kernel_obj"] = "point_interpolator.NadWatIsoSpline(kernel_scale={}, exp_order=2)".format(
        kernel_scale)

    rigid_aug_settings = aug_settings[
        ("rigid_aug", {}, "settings for rigid augmentation")]
    rigid_aug_settings["rotation_range"] = [-15, 15]
    rigid_aug_settings["scale_range"] = [0.8, 1.2]
    rigid_aug_settings["translation_range"] = [-0.1, 0.1]

    spline_aug = SplineAug(aug_settings)

    points_aug = aug_settings[
        ("points_aug", {}, "settings for remove or add noise points")]
    points_aug["remove_random_points"] = False
    points_aug["add_random_point_noise"] = False
    points_aug["add_random_weight_noise"] = True
    points_aug["remove_random_points_by_ratio"] = 0.01
    points_aug["add_random_point_noise_by_ratio"] = 0.01
    points_aug["random_weight_noise_scale"] = 0.1
    points_aug["random_noise_raidus"] = 0.1
    points_aug["normalize_weights"] = False
    points_aug["plot"] = False
    point_aug = PointAug(points_aug)

    def _synth(data_dict):
        synth_info = {"aug_settings":aug_settings}
        points, weights = data_dict["points"],data_dict["weights"]
        if aug_settings["do_point_aug"]:
            points, weights, corr_index = point_aug(points, weights)
            synth_info["corr_index"] = corr_index

        if aug_settings["do_local_deform_aug"] or aug_settings["do_spline_aug"]:
            points, weights = spline_aug(points, weights)
        data_dict["points"], data_dict["weights"] = points, weights
        return data_dict, synth_info
    return _synth




if __name__ == "__main__":
    assert shape_type == "pointcloud", "set shape_type = 'pointcloud'  in global_variable.py"
    device = torch.device("cpu") # cuda:0  cpu
    reader_obj = "lung_dataloader_utils.lung_reader()"
    scale = -1  # an estimation of the physical diameter of the lung, set -1 for auto rescaling   #[99.90687, 65.66011, 78.61013]
    normalizer_obj = "lung_dataloader_utils.lung_normalizer(scale={})".format(scale)
    sampler_obj = "lung_dataloader_utils.lung_sampler(method='voxelgrid',scale=0.001)"
    use_local_mount = True
    remote_mount_transfer = lambda x: x.replace("/playpen-raid1", "/home/zyshen/remote/llr11_mount")
    path_transfer = (lambda x: remote_mount_transfer(x))if use_local_mount else (lambda x: x)

    dataset_json_path = "/playpen-raid1/zyshen/data/point_cloud_expri/train/pair_data.json" #home/zyshen/remote/llr11_mount
    dataset_json_path = path_transfer(dataset_json_path)
    pair_name_list, pair_info_list = read_json_into_list(dataset_json_path)
    pair_path_list = [[pair_info["source"]["data_path"], pair_info["target"]["data_path"]] for pair_info in
                      pair_info_list]
    pair_id = 3
    pair_path = pair_path_list[pair_id]
    pair_path = [path_transfer(path) for path in pair_path]
    get_obj_func = get_obj(reader_obj, normalizer_obj, sampler_obj, device,expand_bch_dim=True)
    source, source_interval = get_obj_func(pair_path[0])
    target, target_interval = get_obj_func(pair_path[1])
    source_points, source_weights = source["points"], source["weights"]

    # set deformation
    aug_settings = ParameterDict()
    aug_settings["do_local_deform_aug"] = True
    aug_settings["do_grid_aug"] = True
    aug_settings["plot"] = True

    local_deform_aug = aug_settings[("local_deform_aug",{},"settings for uniform sampling based spline augmentation")]
    local_deform_aug["num_sample"] = 1000
    local_deform_aug["disp_scale"] = 0.05
    kernel_scale = 0.12
    spline_param = "cov_sigma_scale=0.02,aniso_kernel_scale={},eigenvalue_min=0.3,iter_twice=True, fixed=False, leaf_decay=False, is_interp=True".format(kernel_scale)
    local_deform_aug['local_deform_spline_kernel_obj']="point_interpolator.NadWatAnisoSpline(exp_order=2,{})".format(spline_param)


    grid_spline_aug = aug_settings[("grid_spline_aug",{},"settings for grid sampling based spline augmentation")]
    grid_spline_aug["grid_spacing"] = 0.9
    grid_spline_aug["disp_scale"] = 0.1

    rigid_aug_settings = aug_settings[("rigid_aug", {}, "settings for rigid augmentation")]
    rigid_aug_settings["rotation_range"] = [-15, 15]
    rigid_aug_settings["scale_range"] = [0.8, 1.2]
    rigid_aug_settings["translation_range"] = [-0.1, 0.1]


    kernel_scale = 0.1
    grid_spline_aug["grid_spline_kernel_obj"] = "point_interpolator.NadWatIsoSpline(kernel_scale={}, exp_order=2)".format(kernel_scale)

    st = time.time()
    spline_aug = SplineAug(aug_settings)
    print("it takes {} s".format(time.time()-st))
    spline_aug(source_points,source_weights)

    points_aug = aug_settings[
        ("points_aug", {}, "settings for remove or add noise points")]
    points_aug["remove_random_points"] = False
    points_aug["add_random_point_noise"] = True
    points_aug["add_random_weight_noise"] = True
    points_aug["remove_random_points_by_ratio"] = 0.01
    points_aug["add_random_point_noise_by_ratio"] = 0.01
    points_aug["random_weight_noise_scale"] = 0.05
    points_aug["random_noise_raidus"] = 0.1
    points_aug["normalize_weights"] = False
    points_aug["plot"] = True

    point_aug = PointAug(points_aug)
    point_aug(source_points,source_weights)

