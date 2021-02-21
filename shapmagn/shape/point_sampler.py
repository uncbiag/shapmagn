from pykeops.torch.cluster import grid_cluster
import numpy as np
import torch
from torch_scatter import scatter
from random import Random
import time

def grid_sampler(scale):
    """
    :param scale: voxelgrid gather the point info inside grids of "scale" size
    :return:
    """
    def sampling(points, weights=None):
        """
        :param points: NxD tensor
        :param weights: Nx1 tensor
        :return:
        """
        if weights==None:
            weights = torch.ones(points.shape[0],1).to(points.device)
        index = grid_cluster(points, scale).long()
        points = scatter(points * weights, index, dim=0)
        cluster_weights = scatter(weights, index, dim=0)
        points = points/cluster_weights
        return points, cluster_weights, index
    return sampling

def uniform_sampler(num_sample,fixed_random_seed=True,sampled_by_weight=True):
    """
    :param num_sample: float
    :param rand_generator:
    :return:
    """

    def sampling(points, weights=None):
        """
        :param points:  NxD tensor
        :return:
        """
        if fixed_random_seed:
            np.random.seed(0)
        else:
            np.random.seed(int(time.time()))

        if weights is None:
            weights = torch.ones(points.shape[0],1).to(points.device)
        npoints = points.shape[0]
        if sampled_by_weight:
            weights_np = weights.squeeze().detach().cpu().numpy()
            rand_ind = np.random.choice(np.arange(npoints),num_sample,replace=False,p=weights_np/weights_np.sum())
        else:
            rand_ind = list(range(npoints))
            np.random.shuffle(rand_ind)
            rand_ind = rand_ind[: num_sample]
        rand_ind.sort()
        points = points[rand_ind]
        weights = weights[rand_ind]
        return points, weights, rand_ind
    return sampling


def point_grid_sampler(scale):
    """
    :param scale: voxelgrid gather the point info inside grids of "scale" size
    the batch size typically should be set to 1
    :return:
    """
    grid_point_sampler = grid_sampler(scale)
    def sampling(input_shape):
        from shapmagn.global_variable import Shape
        nbatch = input_shape.nbatch
        sampled_points_list = []
        sampled_weights_list = []
        sampled_pointfea_list = []
        for i in range(nbatch):
            points = input_shape.points[i]
            weights = input_shape.weights[i]
            sampled_points, sampled_weights, index =grid_point_sampler(points, weights)
            if input_shape.pointfea is not None:
                sampled_pointfea = scatter(input_shape.pointfea[i]*weights,index, dim=0)
                sampled_pointfea = sampled_pointfea/sampled_weights
                sampled_pointfea_list.append(sampled_pointfea)
            sampled_points_list.append(sampled_points)
            sampled_weights_list.append(sampled_weights)
        sampled_batch_points = torch.stack(sampled_points_list,dim=0)
        sampled_batch_weights = torch.stack(sampled_weights_list,dim=0)
        # todo for polyline and mesh, edges sampling are not supported
        new_shape = Shape()
        new_shape.set_data_with_refer_to(sampled_batch_points, input_shape)
        new_shape.set_weights(sampled_batch_weights)
        new_shape.set_scale(scale)
        if input_shape.pointfea is not None:
            sampled_batch_pointfea = torch.stack(sampled_pointfea_list,dim=0)
            new_shape.set_pointfea(sampled_batch_pointfea)
        return new_shape
    return sampling


def point_uniform_sampler(num_sample,fixed_random_seed=True, sampled_by_weight=True):
    """
    :param num_sample: float
    :param rand_generator:
    :return:
    """

    uniform_point_sampler = uniform_sampler(num_sample,fixed_random_seed, sampled_by_weight)
    def sampling(input_shape):
        from shapmagn.global_variable import Shape
        nbatch = input_shape.nbatch
        sampled_points_list = []
        sampled_weights_list = []
        sampled_pointfea_list = []
        for i in range(nbatch):
            points = input_shape.points[i]
            weights = input_shape.weights[i]
            sampled_points, sampled_weights, index =uniform_point_sampler(points, weights)
            sampled_weights = sampled_weights
            if input_shape.pointfea is not None:
                sampled_pointfea =input_shape.pointfea[i][index]
                sampled_pointfea_list.append(sampled_pointfea)
            sampled_points_list.append(sampled_points)
            sampled_weights_list.append(sampled_weights)
        sampled_batch_points = torch.stack(sampled_points_list,dim=0)
        sampled_batch_weights = torch.stack(sampled_weights_list,dim=0)
        # todo for polyline and mesh, edges sampling are not supported
        new_shape = Shape()
        new_shape.set_data_with_refer_to(sampled_batch_points, input_shape)
        new_shape.set_weights(sampled_batch_weights)
        new_shape.set_scale(num_sample)
        if input_shape.pointfea is not None:
            sampled_batch_pointfea = torch.stack(sampled_pointfea_list,dim=0)
            new_shape.set_pointfea(sampled_batch_pointfea)
        return new_shape
    return sampling


