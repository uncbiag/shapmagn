"""
data reader for the body.ply
given a file path, the reader will return a dict
{"points":Nx3, "pointfea": NxFeaDim, "weights":Nx1, "type":"pointcloud"}
"""
from random import Random
from shapmagn.datasets.vtk_utils import read_vtk, save_vtk
import numpy as np
import torch
from torch_scatter import scatter
from shapmagn.shape.point_sampler import grid_sampler, uniform_sampler
from shapmagn.shape.shape_utils import get_scale_and_center
from shapmagn.datasets.data_utils import compute_interval

"""
attri points with size (82550, 3)
attri faces with size (100038, 3)
attri RGBA with size (100038, 4)
attri Normals with size (82550, 3)
attri TCoords with size (82550, 2)
"""


def body_reader():
    """
    :return:
    """
    reader = read_vtk
    fea_to_merge = ["TCoords"]
    exp_dim_fn = lambda x: x[:, None] if len(x.shape) == 1 else x

    def norm_fea(fea):
        fea = (fea - fea.mean()) / (fea.std())
        return fea

    def read(file_info):
        path = file_info["data_path"]
        raw_data_dict = reader(path)
        data_dict = {}
        num_points = raw_data_dict["points"].shape[0]
        data_dict["points"] = raw_data_dict["points"]
        data_dict["faces"] = raw_data_dict["faces"]
        data_dict["weights"] = (np.ones([num_points, 1]) / num_points).astype(
            np.float32
        )
        fea_list = [
            norm_fea(exp_dim_fn(raw_data_dict[fea_name])) for fea_name in fea_to_merge
        ]
        data_dict["pointfea"] = np.concatenate(fea_list, 1)
        return data_dict

    return read


def body_sampler(method="uniform", **args):
    """

    :param num_sample: num of points after sampling
    :param method: 'uniform' / 'voxelgrid'
    :param args:
    :return:
    """
    local_rand = Random(0)

    def uniform_sample(data_dict):
        num_sample = args["num_sample"]
        if num_sample != -1:
            points = data_dict["points"]
            weights = data_dict["weights"]
            pointfea = data_dict["pointfea"]
            sampler = uniform_sampler(num_sample, local_rand)
            sampled_points, sampled_weights, ind = sampler(points, weights)
            data_dict["points"] = sampled_points
            data_dict["weights"] = sampled_weights / sum(sampled_weights)
            data_dict["pointfea"] = pointfea[ind]
        return data_dict

    def voxelgrid_sample(data_dict):
        scale = args["scale"]
        if scale != -1:
            points = torch.Tensor(data_dict["points"])
            weights = torch.Tensor(data_dict["weights"])
            pointfea = torch.Tensor(data_dict["pointfea"])
            sampler = grid_sampler(scale)
            points, cluster_weights, index = sampler(points, weights)
            # here we assume the pointfea is summable
            pointfea = scatter(pointfea * weights, index, dim=0)
            pointfea = pointfea / cluster_weights
            # otherwise random sample one from each voxel grid
            # todo complete random sample code by unique sampling from index
            data_dict["points"] = points.numpy()
            data_dict["weights"] = cluster_weights.numpy()
            data_dict["pointfea"] = pointfea.numpy()
        return data_dict

    assert method in [
        "uniform",
        "voxelgrid",
    ], "Not in supported sampler: 'uniform' / 'voxelgrid'"
    sampler = uniform_sample if method == "uniform" else voxelgrid_sample

    def sample(data_dict):
        return sampler(data_dict)

    return sample


def body_normalizer(**args):
    """
    (points-shift)/scale
    :param normalize_coord: bool, normalize coord according to given scale and shift
    :param args: a dict include "scale" : [scale_x,scale_y,scale_z]: , "shift":[shift_x. shift_y, shift_z]
    :return:
    """

    def normalize(data_dict):
        if "scale" in args and "shift" in args:
            scale = np.array(args["scale"])[None]
            shift = np.array(args["shift"])[None]
        else:
            scale, shift = get_scale_and_center(data_dict["points"], percentile=95)
            scale = np.max(scale)
        points = data_dict["points"]
        weights = data_dict["weights"]
        data_dict["points"] = (points - shift) / scale
        data_dict["weights"] = weights / weights.sum()
        return data_dict

    return normalize


if __name__ == "__main__":
    from shapmagn.utils.obj_factory import obj_factory

    reader_obj = "body_dataset_utils.body_reader()"
    # sampler_obj = "lung_utils.lung_sampler(method='uniform',num_sample=1000)"
    sampler_obj = (
        "body_dataset_utils.body_sampler(num_sample=-1, method='voxelgrid',scale=-1)"
    )
    normalizer_obj = "body_dataset_utils.body_normalizer()"
    reader = obj_factory(reader_obj)
    normalizer = obj_factory(normalizer_obj)
    sampler = obj_factory(sampler_obj)
    file_path = "/playpen-raid1/zyshen/debug/body_registration/source.ply"
    file_info = {"name": "body", "data_path": file_path}

    #####################  normalize and save into obj ######################
    raw_data_dict = reader(file_info)
    normalized_data_dict = normalizer(raw_data_dict)
    from shapmagn.utils.shape_visual_utils import save_shape_into_file
    import os

    output_path = "/playpen-raid1/zyshen/debug/body_registration/normalized"
    os.makedirs(output_path, exist_ok=True)
    save_shape_into_file(output_path, "source", ftype="ply", **normalized_data_dict)
    file_path = "/playpen-raid1/zyshen/debug/body_registration/target.ply"
    file_info = {"name": "body", "data_path": file_path}
    raw_data_dict = reader(file_info)
    normalized_data_dict = normalizer(raw_data_dict)
    save_shape_into_file(output_path, "target", ftype="ply", **normalized_data_dict)

    ##################################################################

    raw_data_dict = reader(file_info)
    normalized_data_dict = normalizer(raw_data_dict)
    sampled_data_dict = sampler(normalized_data_dict)
    compute_interval(sampled_data_dict["points"])
    file_path = "/playpen-raid1/zyshen/debug/body_registration/source_norm.vtk"
    save_vtk(file_path, sampled_data_dict)
    file_path = "/playpen-raid1/zyshen/debug/body_registration/target.ply"
    file_info = {"name": "body", "data_path": file_path}
    raw_data_dict = reader(file_info)
    normalized_data_dict = normalizer(raw_data_dict)
    sampled_data_dict = sampler(normalized_data_dict)
    file_path = "/playpen-raid1/zyshen/debug/body_registration/target_norm.vtk"
    save_vtk(file_path, sampled_data_dict)
    compute_interval(sampled_data_dict["points"])
