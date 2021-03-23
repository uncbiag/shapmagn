import os
import torch
import numpy as np
from shapmagn.utils.utils import memory_sort, memory_sort_helper
from shapmagn.shape.point_sampler import uniform_sampler

DEPTH_THRESHOLD = 35.

def flying3d_nonocc_reader():
    def read(file_info):
        path = file_info["data_path"]
        pc = np.load(path)
        pc[..., 0] *= -1
        pc[..., -1] *= -1
        data_dict = {}
        data_dict["points"] = pc
        npoint_source = pc.shape[0]
        data_dict["extra_info"] = {}
        data_dict["extra_info"]["mask"] = np.ones([npoint_source, 1])
        return data_dict
    return read

def flying3d_nonocc_normalizer():
    def normalize(data_dict):
        return data_dict
    return normalize


def flying3d_nonocc_sampler(eps=0,**args):
    def uniform_sample(data_dict, ind=None, fixed_random_seed=True):
        num_sample = args["num_sample"]
        if num_sample != -1:
            points = data_dict["points"]
            mask = data_dict["extra_info"]["mask"].astype(bool)
            if ind is None:
                sampler = uniform_sampler(num_sample, fixed_random_seed,sampled_by_weight=False)
                sampled_points,_, ind = sampler(torch.tensor(points))
                sampled_points = sampled_points.numpy()
                ind = ind.numpy()
            else:
                sampled_points = points[ind]
            data_dict["points"] = sampled_points
            data_dict["extra_info"]["mask"] = mask[ind]
            data_dict["weights"] = np.ones([num_sample,1],dtype=np.float32)/num_sample
            if eps>0:
                data_dict["points"], m_ind = memory_sort(sampled_points,eps)
                data_dict["extra_info"]["mask"] =memory_sort_helper(data_dict["extra_info"]["mask"], m_ind)
        return data_dict, ind
    return uniform_sample


def flying3d_nonocc_pair_postprocess():
    def postprocess(source_dict, target_dict):
        npoints = source_dict["points"].shape[0]
        near_mask = np.logical_and(source_dict["points"][:,2] < DEPTH_THRESHOLD,target_dict["points"][:,2] < DEPTH_THRESHOLD)
        indices = np.where(near_mask)[0]
        if len(indices) <npoints :
            print('{} points deeper than {} has been removed'.format(npoints-len(indices),DEPTH_THRESHOLD))
        extra_info = {key: item[indices] for key, item in source_dict["extra_info"].items()}
        source_dict = {key: item[indices] for key, item in source_dict.items() if key!="extra_info"}
        source_dict.update({"extra_info":extra_info})

        extra_info = {key: item[indices] for key, item in target_dict["extra_info"].items()}
        target_dict = {key: item[indices] for key, item in target_dict.items() if key!="extra_info"}
        target_dict.update({"extra_info":extra_info})


        return source_dict, target_dict
    return postprocess