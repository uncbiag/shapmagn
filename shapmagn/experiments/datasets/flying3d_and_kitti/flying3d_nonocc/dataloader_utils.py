import os
import torch
import numpy as np
from shapmagn.utils.utils import memory_sort
from shapmagn.shape.point_sampler import uniform_sampler

def flyging3d_nonocc_reader():
    def read(file_info):
        path = file_info["data_path"]
        pc = np.load(path)
        pc[..., 0] *= -1
        pc[..., -1] *= -1
        data_dict = {}
        data_dict["points"] = pc
        return data_dict
    return read

def flying3d_nonocc_normalizer():
    def normalize(data_dict):
        return data_dict
    return normalize


def flying3d_nonocc_sampler(**args):
    def uniform_sample(data_dict, fixed_random_seed=True):
        num_sample = args["num_sample"]
        if num_sample != -1:
            points = data_dict["points"]
            gt_mask = data_dict["gt_mask"]
            gt_flow = data_dict["gt_flow"]
            sampler = uniform_sampler(num_sample, fixed_random_seed,sampled_by_weight=False)
            sampled_points, ind = sampler(torch.tensor(points))
            data_dict["points"] = sampled_points.numpy()
            data_dict["gt_mask"] = gt_mask[ind].numpy()
            data_dict["gt_flow"] = gt_flow[ind]
        return data_dict
    return uniform_sample


def flying3d_nonocc_pair_postprocess():
    def postprocess(source_dict, target_dict):
        npoint_source = source_dict["points"].shape[0]
        source_dict["gt_mask"] = np.ones_like(npoint_source,1)
        source_dict["points"],ind = memory_sort(source_dict["points"])
        target_dict["points"] = target_dict["points"][ind]
        source_dict["gt_flow"] = target_dict["points"] - source_dict["points"]
        target_dict["gt_mask"] = source_dict["gt_mask"]
        target_dict["gt_flow"] = source_dict["gt_flow"]
        return source_dict, target_dict
    return postprocess