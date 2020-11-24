
from pykeops.torch import LazyTensor

##################  Lazy Tensor  #######################

class LazyKeopsKernel(object):
    """
    LazyTensor formulaton in Keops,  support batch
    """
    def __init__(self, kernel_type="gauss", **kernel_args):
        assert kernel_type in ["gauss","multi_gauss", "gauss_lin", "gauss_grad", "gauss_lin"]
        self.kernel_type = kernel_type
        self.kernels = {"gauss": self.gauss_kernel,
                        "multi_gauss": self.multi_gauss_kernel,
                        "gauss_grad": self.gaussian_gradient,
                        "gauss_lin":self.gauss_lin_kernel}
        self.kernel = self.kernels[self.kernel_type](**kernel_args)

    @staticmethod
    def gauss_kernel(sigma=0.1):
        """
        :param sigma: scalar
        :return:
        """
        gamma = 1 / (sigma * sigma)

        def reduce(x, y, b):
            """

            :param x: torch.Tensor, BxNxD,  input position1
            :param y: torch.Tensor, BxKxD input position2
            :param b: torch.Tensor, BxKxd, input val
            :return:torch.Tensor, BxNxd, output
            """
            x = LazyTensor(x[:, :, None]) # BxNx1xD
            y = LazyTensor(y[:, None]) # Bx1xKxD
            b = LazyTensor(b[:, None]) # Bx1xKxd
            dist2 = x.sqdist(y)
            kernel = (-dist2 * gamma).exp() # BxNxK
            return (kernel * b).sum_reduction(axis=2)

        return reduce

    @staticmethod
    def multi_gauss_kernel(sigma_list=None, weight_list=None):
        """
        :param sigma_list: a list of sigma
        :param weight_list: corresponding list of weight, sum(weight_list)=1
        :return:
        """
        gamma_list = [1 / (sigma * sigma) for sigma in sigma_list]

        def reduce(x, y, b):
            """

            :param x: torch.Tensor, BxNxD,  input position1
            :param y: torch.Tensor, BxKxD input position2
            :param b: torch.Tensor, BxKxd, input val
            :return:torch.Tensor, BxNxd, output
            """
            kernel=0
            x = LazyTensor(x[:, :, None])
            y = LazyTensor(y[:, None])
            b = LazyTensor(b[:, None])
            dist2 = x.sqdist(y) # BxNxK
            for gamma, weight in zip(gamma_list, weight_list):
                kernel += (-dist2 * gamma).exp() *weight
            return (kernel * b).sum_reduction(axis=2)
        return reduce

    @staticmethod
    def gaussian_gradient(sigma=0.1):
        gamma = 1 / (sigma * sigma)
        def reduce(px, x, py=None, y=None):
            """
           :param px: torch.Tensor, BxNxD,  input position1
           :param x: torch.Tensor, BxNxD input position2
           :param y: torch.Tensor, BxKxD, input val1
           :param py: torch.Tensor, BxKxD, input val2
           :return: torch.Tensor, BxNxD, output
           """
            if y is None:
                y = x
            if py is None:
                py = px
            x = LazyTensor(x[:, :, None]) # BxNx1xD
            y = LazyTensor(y[:, None]) # Bx1xKxD
            px = LazyTensor(px[:,:,None]) # BxNx1xD
            py = LazyTensor(py[:,None,:]) # Bx1xKxD
            dist2 = x.sqdist(y)  # BxNxK
            kernel = (-dist2 * gamma).exp()
            diff_kernel = (x-y) * kernel # BxNxKxD
            pyx = (py*px).sum() #BxNxK
            return (-2 * gamma) * (diff_kernel * pyx).sum_reduction(axis=2)
        return reduce


    @staticmethod
    def gauss_lin_kernel(sigma=0.1):
        """
         :param sigma: scalar
         :return:
         """
        gamma = 1 / (sigma * sigma)
        def reduce(x,y,u,v,b):
            """
            :param x: torch.Tensor, BxNxD,  input position1
            :param y: torch.Tensor, BxKxD input position2
            :param u: torch.Tensor, BxNxD, input val1
            :param v: torch.Tensor, BxKxD, input val2
            :param b: torch.Tensor, BxKxd, input scalar vector
            :return: torch.Tensor, BxNxd, output
            """
            x = LazyTensor(x[:,:, None])
            y = LazyTensor(y[:, None])
            u = LazyTensor(u[:,:,None])
            v = LazyTensor(v[:,None])
            b = LazyTensor(b[:,None]) #Bx1xKxd
            dist2 = x.sqdist(y)
            kernel = (-dist2 * gamma).exp() * ((u | v).square()) #BxNxKx1
            return (kernel * b).sum_reduction(axis=2)
        return reduce

    def __call__(self, *data_args):
        return self.kernel(*data_args)





