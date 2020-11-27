from pykeops.torch.cluster import grid_cluster
import torch
from torch_scatter import scatter
from random import Random
import time
from pykeops.torch import LazyTensor

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
        return points, ind
    return sampling


def kernel_interpolator(scale=0.1, exp_order=2):
    """
    Nadaraya-Watson kernel interpolation

    :param scale: kernel width
    :param exp_order: float, 1,2,0.5
    """
    #todo write plot-test on this function

    assert exp_order in [1,2,0.5]
    def sampling(points,control_points,control_value,control_weights):
        """

        :param points: NxD Tensor
        :param control_points: MxD Tensor
        :param control_value: Mxd Tensor
        :param control_weights: Mx1 Tensor
        :return: Nxd Tensor
        """

        points, control_points = points / scale, control_points / scale

        points_i = LazyTensor(points[:, None, :])  # (N, 1, D)  "column"
        control_points_j = LazyTensor(control_points[None, :, :])  # (1, M, D)  "line"

        dist2 = ((points_i - control_points_j) ** 2).sum(-1)  # (N, M) squared distances

        if exp_order == 1:
            C_ij = dist2.sqrt()  # + D_ij
        elif exp_order == 2:
            C_ij = dist2 / 2
        elif exp_order == 1 / 2:
            C_ij = 2 * dist2.sqrt().sqrt()

        # For the sake of numerical stability, we perform the weight normalization
        # in the log-domain --------------------------------------------------------
        logw_j = LazyTensor(control_weights.log()[None, :, :])

        scores = - (logw_j - C_ij).logsumexp(dim=1)  # (N, 1)
        scores_i = LazyTensor(scores[:, None, :])  # (N, 1, 1)

        value_weight_j = LazyTensor((control_value * control_weights)[None, :, :])  # (1, M, D)

        points_value = ((scores_i - C_ij).exp() * value_weight_j).sum(dim=1)
        return points_value
    return sampling


def spline_intepolator(scale=5, kernel="gauss"):
    """
    Performs a ridge kernel regression, while factoring out affine transforms.

    :param scale: kernel width
    :param kernel: kernel_type, "TPS", "cauchy", "gauss"
    :return:
    """

    # todo write plot-test on this function

    def kernel_matrix(x, y):
        """Implement your favorite SPD kernel matrix here."""
        if kernel != "TPS":
            x, y = x / scale, y / scale

        x_i = LazyTensor(x[:, None, :])  # (N, 1, D)  "column"
        y_j = LazyTensor(y[None, :, :])  # (1, N, D)  "line"

        D_ij = ((x_i - y_j) ** 2).sum(-1)  # (N, N) squared distances

        if kernel == "TPS":  # Thin plate spline in 3D
            K_ij = - D_ij.sqrt()
        elif kernel == "cauchy":
            K_ij = 1 / (1 + D_ij)
        else:  # Gaussian kernel
            K_ij = (- D_ij / 2).exp()  # (N, N)  kernel matrix

        return K_ij  # (N, N) kernel matrix

    def sampling(points, control_points, control_weights, control_value):
        """

        :param points: NxD Tensor
        :param control_points: MxD Tensor
        :param control_value: Mxd Tensor
        :param control_weights: Mx1 Tensor
        :return: Nxd Tensor
        """
        # Sinv(y) = (Id + diag(w) * K_xx)^-1 @ y
        K_xx = kernel_matrix(control_points, control_points)
        wK_xx = LazyTensor(control_weights[:, :, None]) * K_xx
        Sinv = lambda y: wK_xx.solve(y, alpha=1)
        momentum = Sinv(control_value * control_weights)

        # Apply the spline deformation on the full point cloud:
        K_px = kernel_matrix(points, control_points)
        return  K_px @ momentum
    return sampling