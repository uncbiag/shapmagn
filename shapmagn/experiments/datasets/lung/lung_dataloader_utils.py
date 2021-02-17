"""
data reader for the lung.vtk
given a file path, the reader will return a dict
{"points":Nx3, "pointfea": NxFeaDim, "weights":Nx1, "type":"pointcloud"}
"""
from random import Random
from shapmagn.datasets.vtk_utils import read_vtk
import numpy as np
import torch
from torch_scatter import scatter
from shapmagn.shape.point_sampler import grid_sampler, uniform_sampler
from shapmagn.shape.shape_utils import get_scale_and_center
from shapmagn.datasets.data_utils import compute_interval
from shapmagn.experiments.datasets.lung.lung_data_aug import *

"""
attri points with size (73412, 3)
attri scale with size (73412,)
attri hevec1 with size (73412, 3)
attri hevec0 with size (73412, 3)
attri val with size (73412,)
attri hevec2 with size (73412, 3)
attri h0 with size (73412,)
attri h1 with size (73412,)
attri h2 with size (73412,)
attri hess with size (73412, 9)
attri hmode with size (73412,)
attri ChestRegionChestType with size (73412,)
attri dnn_radius with size (73412,)
"""

def lung_reader():
    """
    :return:
    """
    reader = read_vtk
    fea_to_merge = ['points']
    exp_dim_fn = lambda x: x[:,None] if len(x.shape) == 1 else x
    def norm_fea(fea):
        fea = (fea-fea.mean())/(fea.std())
        return fea
    def read(file_info):
        path = file_info["data_path"]
        raw_data_dict = reader(path)
        data_dict = {}
        data_dict["points"] = raw_data_dict["points"]
        data_dict["weights"] = raw_data_dict["dnn_radius"][:,None]
        fea_list = [norm_fea(exp_dim_fn(raw_data_dict[fea_name])) for fea_name in fea_to_merge]
        data_dict["pointfea"] = np.concatenate(fea_list,1)
        return data_dict
    return read



def lung_sampler(method="uniform", **args):
    """

    :param num_sample: num of points after sampling
    :param method: 'uniform' / 'voxelgrid'
    :param args:
    :return:
    """
    local_rand = Random(0)
    def uniform_sample(data_dict):
        num_sample = args["num_sample"]
        if num_sample !=-1:
            points = data_dict["points"]
            weights = data_dict["weights"]
            pointfea = data_dict["pointfea"]
            sampler= uniform_sampler(num_sample, local_rand)
            sampled_points, sampled_weights, ind = sampler(points,weights)
            data_dict["points"] = sampled_points
            data_dict["weights"] = sampled_weights/sum(sampled_weights)
            data_dict["pointfea"] = pointfea[ind]
        return data_dict

    def voxelgrid_sample(data_dict):
        scale = args["scale"]
        if scale != -1:
            points = torch.Tensor(data_dict["points"])
            weights = torch.Tensor(data_dict["weights"])
            pointfea = torch.Tensor(data_dict["pointfea"])
            sampler = grid_sampler(scale)
            points, cluster_weights, index = sampler(points,weights)
            # here we assume the pointfea is summable
            pointfea = scatter(pointfea*weights, index, dim=0)
            pointfea = pointfea / cluster_weights
            # otherwise random sample one from each voxel grid
            # todo complete random sample code by unique sampling from index
            data_dict["points"] = points.numpy()
            data_dict["weights"] = cluster_weights.numpy()
            data_dict["pointfea"] = pointfea.numpy()
        return data_dict

    assert method in ["uniform", "voxelgrid"], "Not in supported sampler: 'uniform' / 'voxelgrid'"
    sampler = uniform_sample if method == "uniform" else voxelgrid_sample
    def sample(data_dict):
        return sampler(data_dict)
    return sample


def lung_normalizer(**args):
    """
    (points-shift)/scale
    :param normalize_coord: bool, normalize coord according to given scale and shift
    :param args: a dict include "scale" : [scale_x,scale_y,scale_z]: , "shift":[shift_x. shift_y, shift_z]
    :return:
    """

    def normalize(data_dict):
        if 'scale' in args and args['scale']!=-1:
            scale = np.array(args['scale'])[None]
            _, shift = get_scale_and_center(data_dict["points"],percentile=95)
        else:
            scale, shift = get_scale_and_center(data_dict["points"],percentile=95)
        points = data_dict["points"]
        weights = data_dict["weights"]
        data_dict["points"] = (points-shift)/scale
        data_dict["weights"] = weights/weights.sum() #/50000
        # data_dict["physical_info"] ={}
        # data_dict["physical_info"] ={'scale':scale,'shift':shift}
        return data_dict
    return normalize




def synth_data(**args):
    aug_settings = ParameterDict()
    aug_settings["do_sampling_aug"] = True
    aug_settings["do_grid_aug"] = True
    aug_settings["do_point_aug"] = False
    aug_settings["plot"] = False
    sampling_spline_aug = aug_settings[
        ("sampling_spline_aug", {}, "settings for uniform sampling based spline augmentation")]
    sampling_spline_aug["num_sample"] = 1000
    sampling_spline_aug["disp_scale"] = 0.05
    kernel_scale = 0.12
    spline_param = "cov_sigma_scale=0.02,aniso_kernel_scale={},eigenvalue_min=0.3,iter_twice=True, fixed=False, leaf_decay=False, is_interp=True".format(
        kernel_scale)
    sampling_spline_aug['sampling_spline_kernel_obj'] = "point_interpolator.NadWatAnisoSpline(exp_order=2,{})".format(
        spline_param)
    grid_spline_aug = aug_settings[("grid_spline_aug", {}, "settings for grid sampling based spline augmentation")]
    grid_spline_aug["grid_spacing"] = 0.9
    grid_spline_aug["disp_scale"] = 0.09
    kernel_scale = 0.1
    grid_spline_aug[
        "grid_spline_kernel_obj"] = "point_interpolator.NadWatIsoSpline(kernel_scale={}, exp_order=2)".format(
        kernel_scale)
    spline_aug = SplineAug(grid_spline_aug)



    points_aug = aug_settings[
        ("points_aug", {}, "settings for remove or add noise points")]
    points_aug["remove_random_points_by_ratio"] = 0.01
    points_aug["add_random_noise_by_ratio"] = 0.01
    points_aug["random_noise_raidus"] = 0.1
    points_aug["normalize_weights"] = False
    points_aug["plot"] = False
    point_aug = PointAug(points_aug)

    def _synth(data_dict):
        synth_info = {"aug_settings":aug_settings}
        points, weights = torch.Tensor(data_dict["points"]),torch.Tensor(data_dict["weights"])
        if aug_settings["do_point_aug"]:
            points, weights, corr_index = point_aug(points, weights)
            synth_info["corr_index"] = corr_index

        if aug_settings["do_sampling_aug"]  or aug_settings["do_spline_aug"]:
            points, weights = spline_aug(points, weights)
        data_dict["points"], data_dict["weights"] = points.numpy(), weights.numpy()
        return data_dict, synth_info
    return _synth




if __name__ == "__main__":
    from shapmagn.utils.obj_factory import obj_factory
    reader_obj = "lung_dataloader_utils.lung_reader()"
    #sampler_obj = "lung_utils.lung_sampler(method='uniform',num_sample=1000)"
    sampler_obj = "lung_dataloader_utils.lung_sampler(method='voxelgrid',scale=-1)"
    normalizer_obj = "lung_dataloader_utils.lung_normalizer()"
    reader = obj_factory(reader_obj)
    normalizer = obj_factory(normalizer_obj)
    sampler = obj_factory(sampler_obj)
    file_path = "/playpen-raid1/Data/UNC_vesselParticles/10005Q_EXP_STD_NJC_COPD_wholeLungVesselParticles.vtk"
    file_info = {"name":file_path,"data_path":file_path}
    raw_data_dict  = reader(file_info)
    normalized_data_dict = normalizer(raw_data_dict)
    sampled_data_dict = sampler(normalized_data_dict)
    compute_interval(sampled_data_dict["points"])


