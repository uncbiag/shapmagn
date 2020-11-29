"""
data reader for the lung.vtk
given a file path, the reader will return a dict
{"points":Nx3, "pointfea": NxFeaDim, "weights":Nx1, "type":"pointcloud"}
"""
from random import Random
from shapmagn.datasets.vtk_utils import read_vtk
import numpy as np
from pykeops.torch.cluster import grid_cluster
import torch
from torch_scatter import scatter
from shapmagn.utils.sampler import grid_sampler, uniform_sampler

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
    fea_to_merge = ['points', 'scale']
    exp_dim_fn = lambda x: x[:,None] if len(x.shape) == 1 else x
    def read(file_info):
        path = file_info["data_path"]
        raw_data_dict = reader(path)
        data_dict = {}
        data_dict["points"] = raw_data_dict["points"]
        data_dict["weights"] = raw_data_dict["scale"][:,None]
        fea_list = [exp_dim_fn(raw_data_dict[fea_name]) for fea_name in fea_to_merge]
        data_dict["pointfea"] = np.concatenate(fea_list,1)
        return data_dict
    return read



def lung_sampler(num_sample=-1, method="uniform", **args):
    """

    :param num_sample: num of points after sampling
    :param method: 'uniform' / 'voxelgrid'
    :param args:
    :return:
    """
    local_rand = Random(0)
    def uniform_sample(data_dict):
        points = data_dict["points"]
        weights = data_dict["weights"]
        pointfea = data_dict["pointfea"]
        sampler= uniform_sampler(num_sample, local_rand)
        sampled_points, ind = sampler(points)
        data_dict["weights"] = weights[ind]
        data_dict["pointfea"] = pointfea[ind]
        return data_dict

    def voxelgrid_sample(data_dict):
        scale = args["scale"]
        points = torch.Tensor(data_dict["points"])
        weights = torch.Tensor(data_dict["weights"])
        pointfea = torch.Tensor(data_dict["pointfea"])
        sampler = grid_sampler(scale)
        points, weights, index = sampler(points,weights)
        points = points / weights
        # here we assume the pointfea is summable
        pointfea = scatter(pointfea, index, dim=0)
        pointfea = pointfea / weights
        # otherwise random sample one from each voxel grid
        # todo complete random sample code by unique sampling from index
        data_dict["points"] = points.numpy()
        data_dict["weights"] = weights.numpy()
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

    def get_scale_and_center(points,percentile=0.99):
        dim = points.shape[1]
        scale = np.zeros(1,dim)
        center = np.zeros(1,dim)
        interval = [(1-percentile)/2,1-(1-percentile)/2]
        for d in range(dim):
            filtered_low_thre =  np.percentile(points[:,d],interval[0])
            filtered_up_thre =  np.percentile(points[:,d],interval[1])
            scale[0,d] = filtered_up_thre - filtered_low_thre
            center[0,d] = (filtered_up_thre + filtered_low_thre)/2
        return scale, center

    def normalize(data_dict):
        if 'scale' in args and 'shift' in args:
            scale = np.array(args['scale'])[None]
            shift = np.array(args['shift'])[None]
        else:
            scale, shift = get_scale_and_center(data_dict["points"],percentile=0.95)
        points = data_dict["points"]
        weights = data_dict["weights"]
        data_dict["points"] = (points-shift)/scale
        data_dict["weights"] = weights/weights.sum()
        data_dict["physical_info"] ={}
        data_dict["physical_info"] ={'scale':scale,'shift':shift}
        return data_dict
    return normalize



if __name__ == "__main__":
    from shapmagn.utils.obj_factory import obj_factory
    reader_obj = "lung_dataset_utils.lung_reader()"
    #sampler_obj = "lung_utils.lung_sampler(num_sample=1000, method='uniform')"
    sampler_obj = "lung_dataset_utils.lung_sampler(num_sample=1000, method='voxelgrid',scale=5)"
    normalizer_obj = "lung_dataset_utils.lung_normalizer(scale=[100,100,100],shift=[50,50,50])"
    reader = obj_factory(reader_obj)
    sampler = obj_factory(sampler_obj)
    normalizer = obj_factory(normalizer_obj)
    file_path = "/playpen-raid1/Data/UNC_vesselParticles/10005Q_EXP_STD_NJC_COPD_wholeLungVesselParticles.vtk"
    file_info = {"name":file_path,"data_path":file_path}
    raw_data_dict  = reader(file_info)
    sampled_data_dict = sampler(raw_data_dict)
    normalized_data_dict = normalizer(sampled_data_dict)

