from geomloss.utils import distances

##################  Generic Tunction formulation  ######################


class GenericKeopsKernel(object):
    """
    Generic symbol formulation in Keops,  not batch support
    """
    def __init__(self, kernel_type, *kernel_args):
        assert kernel_type in ["gauss","gauss_lin"]
        self.kernel_type = kernel_type
        self.kernels = {"gauss": self.gauss_kernel(*kernel_args), "gauss_lin":self.gauss_lin_kernel(*kernel_args)}

    @staticmethod
    def gauss_kernel(sigma):
        """

        :param sigma: scalar
        :return:
        """
        x, y, b = Vi(0, 3), Vj(1, 3), Vj(2, 3)
        gamma = 1 / (sigma * sigma)
        D2 = x.sqdist(y)
        K = (-D2 * gamma).exp()
        return (K * b).sum_reduction(axis=1)[:,None]

    @staticmethod
    def gauss_lin_kernel(sigma):
        """
        :param sigma: scalar
        :return:
        """
        x, y, u, v, b = Vi(0, 3), Vj(1, 3), Vi(2, 3), Vj(3, 3), Vj(4, 1)
        gamma = 1 / (sigma * sigma)
        D2 = x.sqdist(y)
        K = (-D2 * gamma).exp() * (u * v).sum() ** 2
        return (K * b).sum_reduction(axis=1)[:,None]



    def __call__(self, *data_args):
        return self.kernels[self.kernel_type](*data_args)