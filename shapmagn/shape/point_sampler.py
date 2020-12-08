from pykeops.torch.cluster import grid_cluster
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

def uniform_sampler(num_sample, rand_generator=Random(0)):
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
        if weights==None:
            weights = torch.ones(points.shape[0],1).to(points.device)
        npoints = points.shape[0]
        ind = list(range(npoints))
        rand_generator.shuffle(ind)
        ind = ind[: num_sample]
        # continuous in spatial
        ind.sort()
        points = points[ind]
        weights = weights[ind]
        return points, weights, ind
    return sampling


def grid_shape_sampler(scale):
    """
    :param scale: voxelgrid gather the point info inside grids of "scale" size
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


def uniform_shape_sampler(num_sample, rand_generator=Random(0)):
    """
    :param num_sample: float
    :param rand_generator:
    :return:
    """

    uniform_point_sampler = uniform_sampler(num_sample, rand_generator=rand_generator)
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
            sampled_weights = sampled_weights/sum(sampled_weights)
            if input_shape.pointfea is not None:
                sampled_pointfea =input_shape.pointfea[index]
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


