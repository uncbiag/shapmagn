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
        weights = scatter(weights, index, dim=0)
        points = points/weights
        return points, weights, index
    return sampling
def uniform_sampler(num_sample, rand_generator=Random(int(time.time()))):
    """
    :param num_sample: float
    :param rand_generator:
    :return:
    """
    def sampling(points):
        """
        :param points:  NxD tensor
        :return:
        """
        npoints = points.shape[0]
        ind = list(range(npoints))
        rand_generator.shuffle(ind)
        ind = ind[: num_sample]
        # continuous in spatial
        ind.sort()
        points = points[ind]
        return points, None, ind
    return sampling

