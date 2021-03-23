import os
import torch
import numpy as np
from shapmagn.utils.utils import memory_sort
from shapmagn.shape.point_sampler import uniform_sampler
DEPTH_THRESHOLD=35
def flying3d_hasocc_reader():
    def read(file_info):
        path = file_info["data_path"]
        pc = np.load(path)
        data_dict = {}
        # data_dict["points1"], ind = memory_sort(pc["points1"])
        # data_dict["points2"], _ = memory_sort(pc["points2"])
        data_dict["extra_info"] = {}
        data_dict["extra_info"]["mask"] = pc["valid_mask1"].reshape(-1, 1)
        data_dict["extra_info"]["gt_flow"] = pc["flow"]
        return data_dict
    return read

def flying3d_hasocc_normalizer():
    def normalize(data_dict):
        return data_dict
    return normalize


def flying3d_hasocc_sampler(**args):
    def uniform_sample(data_dict, ind=None, fixed_random_seed=True):
        num_sample = args["num_sample"]
        if num_sample != -1:
            points = data_dict["points"]
            mask = data_dict["extra_info"]["mask"]
            gt_flow = data_dict["extra_info"]["gt_flow"]
            if ind is None:
                sampler = uniform_sampler(num_sample, fixed_random_seed, sampled_by_weight=False)
                sampled_points, ind = sampler(torch.tensor(points))
                sampled_points = sampled_points.numpy()
            else:
                sampled_points = points[ind]
            data_dict["weights"] = np.ones([num_sample,1],dtype=np.float32)/num_sample
            data_dict["points"] = sampled_points
            data_dict["extra_info"]["mask"] = mask[ind].numpy()
            data_dict["extra_info"]["gt_flow"] = gt_flow[ind]
        return data_dict, ind
    return uniform_sample

def flying3d_hasocc_pair_postprocess():
    def postprocess(source_dict, target_dict):
        npoints = source_dict["points"].shape[0]
        near_mask = np.logical_and(source_dict["points"][:, 2] < DEPTH_THRESHOLD,
                                   target_dict["points"][:, 2] < DEPTH_THRESHOLD)
        indices = np.where(near_mask)[0]
        if len(indices) < npoints:
            print('{} points deeper than {} has been removed'.format(npoints - len(indices), DEPTH_THRESHOLD))
        extra_info = {key: item[indices] for key, item in source_dict["extra_info"].items()}
        source_dict = {key: item[indices] for key, item in source_dict.items() if key != "extra_info"}
        source_dict.update({"extra_info": extra_info})

        extra_info = {key: item[indices] for key, item in target_dict["extra_info"].items()}
        target_dict = {key: item[indices] for key, item in target_dict.items() if key != "extra_info"}
        target_dict.update({"extra_info": extra_info})
        return source_dict, target_dict
    return postprocess