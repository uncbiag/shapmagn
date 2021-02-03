import torch
from pykeops.torch import LazyTensor

# def nadwat_kernel_interpolator(scale=0.1, exp_order=2):
#     """
#     Nadaraya-Watson kernel interpolation
#
#     :param scale: kernel width
#     :param exp_order: float, 1,2,0.5
#     """
#     #todo write plot-test on this function
#
#     assert exp_order in [1,2,0.5]
#     def interp(points,control_points,control_value,control_weights):
#         """
#
#         :param points: NxD Tensor
#         :param control_points: MxD Tensor
#         :param control_value: Mxd Tensor
#         :param control_weights: Mx1 Tensor
#         :return: Nxd Tensor
#         """
#
#         points, control_points = points / scale, control_points / scale
#
#         points_i = LazyTensor(points[:, None, :])  # (N, 1, D)  "column"
#         control_points_j = LazyTensor(control_points[None, :, :])  # (1, M, D)  "line"
#
#         dist2 = ((points_i - control_points_j) ** 2).sum(-1)  # (N, M) squared distances
#
#         if exp_order == 1:
#             C_ij = dist2.sqrt()  # + D_ij
#         elif exp_order == 2:
#             C_ij = dist2 / 2
#         elif exp_order == 1 / 2:
#             C_ij = 2 * dist2.sqrt().sqrt()
#
#         # For the sake of numerical stability, we perform the weight normalization
#         # in the log-domain --------------------------------------------------------
#         logw_j = LazyTensor(control_weights.log()[None, :, :])
#
#         scores = - (logw_j - C_ij).logsumexp(dim=1)  # (N, 1)
#         scores_i = LazyTensor(scores[:, None, :])  # (N, 1, 1)
#
#         value_weight_j = LazyTensor((control_value * control_weights)[None, :, :])  # (1, M, D)
#
#         points_value = ((scores_i - C_ij).exp() * value_weight_j).sum(dim=1)
#         return points_value
#     return interp



def nadwat_kernel_interpolator(scale=0.1, exp_order=2,iso=True):
    """
    Nadaraya-Watson kernel interpolation

    :param scale: kernel width of isotropic kernel, disabled if the iso is False
    :param exp_order: float, 1,2,0.5
    :param iso: bool, use isotropic kernel, sigma equals to scale
    """
    #todo write plot-test on this function

    assert exp_order in [1,2,0.5]
    def interp(points,control_points,control_value,control_weights, gamma=None):
        """

        :param points: BxNxD Tensor
        :param control_points: BxMxD Tensor
        :param control_value: BxMxd Tensor
        :param control_weights: BxMx1 Tensor
        :param gamma: optional BxMxDxD Tensor, anisotropic inverse kernel
        :return: BxNxd Tensor
        """


        if iso:
            points, control_points = points / scale, control_points / scale
            points_i = LazyTensor(points[:, :, None, :])  # (B,N, 1, D)  "column"
            control_points_j = LazyTensor(control_points[:, None, :, :])  # (B,1, M, D)  "line"
            dist2 = ((points_i - control_points_j) ** 2).sum(-1)  # (N, M) squared distances (B,N,M,1)
        else:
            points_i = LazyTensor(points[:, :, None, :])  # (B,N, 1, D)  "column"
            control_points_j = LazyTensor(control_points[:, None, :, :])  # (B,1, M, D)  "line"
            gamma = LazyTensor(gamma.view(gamma.shape[0], gamma.shape[1], -1)[:, None])  # Bx1xMxD*D
            dist2 = (points_i - control_points_j) | gamma.matvecmult(points_i - control_points_j)

        if exp_order == 1:
            C_ij = dist2.sqrt()  # + D_ij
        elif exp_order == 2:
            C_ij = dist2 / 2
        elif exp_order == 1 / 2:
            C_ij = 2 * dist2.sqrt().sqrt()

        # For the sake of numerical stability, we perform the weight normalization
        # in the log-domain --------------------------------------------------------
        logw_j = LazyTensor(control_weights.log()[:,None, :, :]) #(B,1,M,1)

        scores = - (logw_j - C_ij).logsumexp(dim=2)  # (B,N, 1)
        scores_i = LazyTensor(scores[:,:, None, :])  # (B,N, 1, 1)

        value_weight_j = LazyTensor((control_value * control_weights)[:,None, :, :])  # (B,1, M, D)

        points_value = ((scores_i - C_ij).exp() * value_weight_j).sum(dim=2)
        return points_value
    return interp




def _spline_intepolator(scale=0.1, kernel="gauss",iso=True):
    """
    Performs a ridge kernel regression.

    :param scale: kernel width
    :param kernel: kernel_type, "TPS", "cauchy", "gauss"
    :return:
    """

    # todo write plot-test on this function

    def kernel_matrix(x, y, gamma=None):
        """Implement your favorite SPD kernel matrix here."""
        if kernel != "TPS" and iso:
            x, y = x / scale, y / scale

        if iso:
            x_i = LazyTensor(x[:, None, :])  # (N, 1, D)  "column"
            y_j = LazyTensor(y[None, :, :])  # (1, N, D)  "line"
            D_ij = ((x_i - y_j) ** 2).sum(-1)  # (N, N) squared distances
        else:
            x_i = LazyTensor(x[:, None, :])  # (N, 1, D)  "column"
            y_j = LazyTensor(y[None, :, :])  # (1, N, D)  "line"
            gamma = LazyTensor(gamma.view(gamma.shape[0], -1)[None])  # 1xNxD*D
            D_ij = (x_i - y_j) | gamma.matvecmult(x_i - y_j)  # (N, N) squared distances

        if kernel == "TPS":  # Thin plate spline in 3D
            K_ij = - D_ij.sqrt()
        elif kernel == "cauchy":
            K_ij = 1 / (1 + D_ij)
        else:  # Gaussian kernel
            K_ij = (- D_ij / 2).exp()  # (N, N)  kernel matrix

        return K_ij  # (N, N) kernel matrix

    def interp(points, control_points, control_weights, control_value, gamma=None):
        """

        :param points: NxD Tensor
        :param control_points: MxD Tensor
        :param control_value: Mxd Tensor
        :param control_weights: Mx1 Tensor
        :return: Nxd Tensor
        """
        # Sinv(y) = (Id + diag(w) * K_xx)^-1 @ y
        assert iso==True, "aniso is not fully supported yet"
        K_xx = kernel_matrix(control_points, control_points)
        wK_xx = LazyTensor(control_weights[:, :, None]) * K_xx
        Sinv = lambda y: wK_xx.solve(y, alpha=1)
        momentum = Sinv(control_value * control_weights)
        # Apply the spline deformation on the full point cloud:
        K_px = kernel_matrix(points, control_points)
        return K_px @ momentum
    return interp






def ridge_kernel_intepolator(scale=0.1, kernel="gauss"):
    """
    Performs a ridge kernel regression,

    :param scale: kernel width
    :param kernel: kernel_type, "TPS", "cauchy", "gauss"
    :return:
    """
    # todo rewrite in batch form
    _interp = _spline_intepolator(scale=scale, kernel=kernel)
    def interp(points, control_points, control_weights, control_value, gamma=None):
        interpolated_list = []
        gamma = gamma if gamma is not None else [None]*points.shape[0]
        for _points, _control_points, _control_weights, _control_value, _gamma in zip(points, control_points, control_weights, control_value, gamma):
            output = _interp( _points, _control_points, _control_weights, _control_value, _gamma)
            interpolated_list.append(output)
        return torch.stack(interpolated_list,1)
    return interp





