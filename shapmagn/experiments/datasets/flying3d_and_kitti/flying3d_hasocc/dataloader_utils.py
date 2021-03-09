import os
import torch
import numpy as np
from shapmagn.utils.utils import memory_sort
from shapmagn.shape.point_sampler import uniform_sampler

def flyging3d_hasocc_reader():
    def read(file_info):
        path = file_info["data_path"]
        pc = np.load(path)
        data_dict = {}
        data_dict["points1"], ind = memory_sort(pc["points1"])
        data_dict["points2"], _ = memory_sort(pc["points2"])
        data_dict["gt_mask"] = pc["valid_mask1"].reshape(-1, 1)[ind]
        data_dict["gt_flow"] = pc["flow"][ind]
        return data_dict
    return read

def flying3d_hasocc_normalizer():
    def normalize(data_dict):
        return data_dict
    return normalize


def flying3d_hasocc_sampler(**args):
    def uniform_sample(data_dict, fixed_random_seed=True):
        num_sample = args["num_sample"]
        if num_sample != -1:
            points = data_dict["points"]
            gt_mask = data_dict["gt_mask"]
            gt_flow = data_dict["gt_flow"]
            sampler = uniform_sampler(num_sample, fixed_random_seed, sampled_by_weight=False)
            sampled_points, ind = sampler(torch.tensor(points))
            data_dict["points"] = sampled_points.numpy()
            data_dict["gt_mask"] = gt_mask[ind].numpy()
            data_dict["gt_flow"] = gt_flow[ind]
        return data_dict
    return uniform_sample

def flying3d_nonocc_pair_postprocess():
    def postprocess(source_dict, target_dict):
        return source_dict, target_dict
    return postprocess