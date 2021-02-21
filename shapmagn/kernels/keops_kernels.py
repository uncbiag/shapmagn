
from pykeops.torch import Vi, Vj, Pm, LazyTensor

##################  Lazy Tensor  #######################

class LazyKeopsKernel(object):
    """
    LazyTensor formulaton in Keops,  support batch
    """
    def __init__(self, kernel_type="gauss", **kernel_args):
        assert kernel_type in ["gauss", "multi_gauss", "gauss_lin", "gauss_grad", "multi_gauss_grad", "gauss_lin"]
        self.kernel_type = kernel_type
        self.kernels = {"gauss": self.gauss_kernel,
                        "normalized_gauss": self.normalized_gauss_kernel,
                        "multi_gauss": self.multi_gauss_kernel,
                        "gauss_grad": self.gaussian_gradient,
                        "multi_gauss_grad": self.multi_gaussian_gradient,
                        "gauss_lin": self.gauss_lin_kernel}
        self.kernel = self.kernels[self.kernel_type](**kernel_args)

    @staticmethod
    def gauss_kernel(sigma=0.1):
        """
        :param sigma: scalar
        :return:
        """
        def conv(x, y, b):
            """

            :param x: torch.Tensor, BxNxD,  input position1
            :param y: torch.Tensor, BxKxD input position2
            :param b: torch.Tensor, BxKxd, input val
            :return:torch.Tensor, BxNxd, output
            """
            x = LazyTensor(x[:, :, None]/sigma) # BxNx1xD
            y = LazyTensor(y[:, None]/sigma) # Bx1xKxD
            b = LazyTensor(b[:, None]) # Bx1xKxd
            dist2 = x.sqdist(y)
            kernel = (-dist2).exp() # BxNxK
            return (kernel * b).sum_reduction(axis=2)

        return conv

    @staticmethod
    def normalized_gauss_kernel(sigma=0.1):
        """
        :param sigma: scalar
        :return:
        """

        def conv(x, y, b):
            """

            :param x: torch.Tensor, BxNxD,  input position1
            :param y: torch.Tensor, BxKxD input position2
            :param b: torch.Tensor, BxKxd, input val
            :return:torch.Tensor, BxNxd, output
            """
            x = LazyTensor(x[:, :, None] / sigma)  # BxNx1xD
            y = LazyTensor(y[:, None] / sigma)  # Bx1xKxD
            b = LazyTensor(b[:, None])  # Bx1xKxd
            dist2 = x.sqdist(y)
            return dist2.sumsoftmaxweight(b,axis=2)
        return conv

    @staticmethod
    def multi_gauss_kernel(sigma_list=None, weight_list=None):
        """
        :param sigma_list: a list of sigma
        :param weight_list: corresponding list of weight, sum(weight_list)=1
        :return:
        """
        gamma_list = [1 / (sigma * sigma) for sigma in sigma_list]


        def conv(x, y, b):
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
        return conv

    @staticmethod
    def gaussian_gradient(sigma=0.1):
        def conv(px, x, py=None, y=None):
            """
           :param px: torch.Tensor, BxNxD,  input position1
            :param y: torch.Tensor, BxKxD, input val1
           :param py: torch.Tensor, BxNxD input position2
           :param y: torch.Tensor, BxKxD, input val2
           :return: torch.Tensor, BxNxD, output
           """
            if y is None:
                y = x
            if py is None:
                py = px
            x = LazyTensor(x[:, :, None]/sigma) # BxNx1xD
            y = LazyTensor(y[:, None]/sigma) # Bx1xKxD
            px = LazyTensor(px[:,:,None]) # BxNx1xD
            py = LazyTensor(py[:,None,:]) # Bx1xKxD
            dist2 = x.sqdist(y)  # BxNxK
            kernel = (-dist2).exp()
            diff_kernel = (x-y) * kernel # BxNxKxD
            pyx = (py*px).sum() #BxNxK
            return (-2/sigma) * (diff_kernel * pyx).sum_reduction(axis=2)
        return conv

    @staticmethod
    def multi_gaussian_gradient(sigma_list=None, weight_list=None):
        """
       :param sigma_list: a list of sigma
       :param weight_list: corresponding list of weight, sum(weight_list)=1
       :return:
        """
        gamma_list = [1 / (sigma * sigma) for sigma in sigma_list]

        def conv(px, x, py=None, y=None):
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
            kernel = 0.
            x = LazyTensor(x[:, :, None])  # BxNx1xD
            y = LazyTensor(y[:, None])  # Bx1xKxD
            px = LazyTensor(px[:, :, None])  # BxNx1xD
            py = LazyTensor(py[:, None, :])  # Bx1xKxD
            dist2 = x.sqdist(y)  # BxNxK
            for gamma, weight in zip(gamma_list, weight_list):
                kernel += ((-dist2 * gamma).exp()) * gamma * weight  # BxNxKx1
            diff_kernel = (x - y) * kernel  # BxNxKxD
            pyx = (py * px).sum()  # BxNxK
            return (-2) * (diff_kernel * pyx).sum_reduction(axis=2)
        return conv


    @staticmethod
    def gauss_lin_kernel(sigma=0.1):
        """
         :param sigma: scalar
         :return:
         """
        def conv(x,y,u,v,b):
            """
            :param x: torch.Tensor, BxNxD,  input position1
            :param y: torch.Tensor, BxKxD input position2
            :param u: torch.Tensor, BxNxD, input val1
            :param v: torch.Tensor, BxKxD, input val2
            :param b: torch.Tensor, BxKxd, input scalar vector
            :return: torch.Tensor, BxNxd, output
            """
            x = LazyTensor(x[:,:, None]/sigma)
            y = LazyTensor(y[:, None]/sigma)
            u = LazyTensor(u[:,:,None])
            v = LazyTensor(v[:,None])
            b = LazyTensor(b[:,None]) #Bx1xKxd
            dist2 = x.sqdist(y)
            kernel = (-dist2).exp() * ((u | v).square()) #BxNxKx1
            return (kernel * b).sum_reduction(axis=2)
        return conv

    def __call__(self, *data_args):
        return self.kernel(*data_args)


    @staticmethod
    def aniso_sp_gauss_kernel(neigh_to_center=True):
        """
        anisotropic rbf kernel

        if neigh_to_center value defined by neigh-to-center sum
        if not neigh_to_center value defined by center-to-neighing sum
        """
        reduce_dim = 2 if neigh_to_center else 1

        def conv(x, gamma, b):
            """

            :param x: torch.Tensor, BxNxD,
            :param gamma: BxNxDxD
            :param b: torch.Tensor, BxNxd, input val
            :return:torch.Tensor, BxNxd, output
            """
            x_i = LazyTensor(x[:, :, None])  # BxNx1xD
            x_j = LazyTensor(x[:, None])  # Bx1xNxD
            b = LazyTensor(b[:, None])  # Bx1xNxd
            gamma = LazyTensor(gamma.view(gamma.shape[0], gamma.shape[1], -1)[:, None])  # Bx1xNxD*D
            dist2 = (x_i - x_j) | gamma.matvecmult(x_i - x_j) # BxNxNxD | Bx1xNxDxD * BxNxNxD    -> BxNxN
            kernel = (-dist2).exp()  # BxNxN
            return kernel.t() @ ((kernel * b).sum_reduction(axis=reduce_dim))
        return conv


    @staticmethod
    def aniso_sp_gauss_interp_kernel():
        """
        anisotropic rbf interpolation kernel,

        """
        reduce_dim = 2
        assert False,"not check yet"
        def conv(x,y, gamma_x,gamma_y, b):
            """

            :param x: torch.Tensor, BxNxD,
            :param y: torch.Tensor, BxKxD,
            :param gamma_x: BxNxDxD
            :param gamma_y: BxKxDxD
            :param b: torch.Tensor, BxKxd, input val
            :return: torch.Tensor, BxNxd, output
            """
            x_i = LazyTensor(x[:, :, None])  # BxNx1xD
            x_j = LazyTensor(x[:,None])  # Bx1xNxD
            y_j = LazyTensor(y[:, None])  # Bx1xKxD
            b = LazyTensor(b[:, None])  # Bx1xKxd
            gamma_x = LazyTensor(gamma_x.view(gamma_x.shape[0], gamma_x.shape[1], -1)[:, None])  # Bx1xNxD*D
            gamma_y = LazyTensor(gamma_y.view(gamma_y.shape[0], gamma_y.shape[1], -1)[:, None])  # Bx1xKxD*D
            dist2_xx = (x_i - x_j) | gamma_x.matvecmult(x_i - x_j)  # BxNxNxD | Bx1xNxDxD * BxNxNxD    -> BxNxN
            dist2_xy = (x_i - y_j) | gamma_y.matvecmult(x_i - y_j)  # BxNxKxD | Bx1xKxDxD * BxNxKxD    -> BxNxK
            kernel_xx = (-dist2_xx).exp()  # BxNxN
            kernel_xy = (-dist2_xy).exp()  # BxNxK
            return kernel_xx.t()@((kernel_xy * b).sum_reduction(axis=reduce_dim))
        return conv



if __name__ == "__main__":
    from shapmagn.kernels.keops_kernels import *
    import torch
    batch_sz = 1
    gamma = torch.rand(3,3).repeat(batch_sz,1500,1,1)
    x = torch.rand(batch_sz,1500,3)
    b = torch.rand(batch_sz,1500,2)
    kernel1 = LazyKeopsKernel.aniso_sp_gauss_kernel()
    kernel2 = LazyKeopsKernel.aniso_sp_gauss_interp_kernel()
    z1 = kernel1(torch.cat([x,x],0),gamma, torch.cat([b,b],0))
    z2 = kernel2(torch.cat([x,x],0),torch.cat([x,x],0),gamma, gamma,torch.cat([b,b],0))
    print()

