"""
the code here is largely borrowed from deformetrica
here turns it into a batch-supported version
"""

import torch
from geomloss import SamplesLoss
from shapmagn.utils.keops_kernels import LazyKeopsKernel
from shapmagn.utils.torch_kernels import TorchKernel


class CurrentDistance(object):
    def __init__(self, opt):
        current_opt = opt[('current',{},"settings for the current loss")]
        kernel_backend = current_opt[("kernel_backend",'torch',"kernel backend can either be 'torch'/'keops'")]
        sigma = current_opt[('sigma',0.1,"the sigma in gaussian kernel")]
        self.kernel = LazyKeopsKernel('gauss',sigma) if kernel_backend == 'keops' else TorchKernel('gauss',sigma)

    def __call__(self,moved, target):
        assert moved.type == 'PolyLine'
        batch = moved.batch
        c1, n1 = moved.get_centers_and_currents()
        c2, n2 = target.get_centers_and_currents()

        def current_scalar_product(points_1, points_2, normals_1, normals_2):
            return  ( (normals_1.view(batch,-1))
                    * (self.kernel(points_1, points_2, normals_2).view(batch,-1))
                    ).sum(-1)

        distance =  current_scalar_product(c1, c1, n1, n1) \
                + current_scalar_product(c2, c2, n2, n2) \
                - 2 * current_scalar_product(c1, c2, n1, n2)
        return distance

class VarifoldDistance(object):
    def __init__(self, opt):
        varifold_opt = opt[('varifold', {}, "settings for the varifold loss")]
        kernel_backend = varifold_opt[("kernel_backend", 'torch', "kernel backend can either be 'torch'/'keops'")]
        sigma = varifold_opt[('sigma', 0.1, "the sigma in gaussian lin kernel")]
        self.kernel = LazyKeopsKernel('gauss_lin', sigma) if kernel_backend == 'keops' else TorchKernel('gauss_lin', sigma)
    def __call__(self, moved, target):
        assert moved.type == 'SurfaceMesh'
        batch = moved.batch
        c1, n1 = moved.get_centers_and_normals()
        c2, n2 = target.get_centers_and_normals()
        areaa = torch.norm(n1, 2, 2) [None]  # BxNx1
        areab = torch.norm(n2, 2, 2)[None]   # BxKx1
        nalpha = n1 / areaa
        nbeta = n2 / areab

        def varifold_scalar_product(x, y, areaa, areab, nalpha, nbeta):
            return ((areaa.view(batch,-1)
                     * (self.kernel(x, y, nalpha, nbeta, areab).view(batch,-1)))
                    ).sum(-1)

        return varifold_scalar_product(c1, c1, areaa, areaa, nalpha, nalpha) \
                +varifold_scalar_product(c2, c2, areab, areab, nbeta, nbeta) \
               - 2 * varifold_scalar_product(c1, c2, areaa, areab, nalpha, nbeta)



class L2Distance(object):
    def __init__(self, opt):
        l2_opt = opt[('l2',{},"settings for the l2 loss")]
        self.attr = l2_opt[('attr','landmarks',"compute distance on the specific class attribute: 'ponts','landmarks','pointfea")]
    def __call__(self,moved, target):
        batch = moved.batch
        attr1 = moved.getattr(self.attr)
        attr2 = target.getattr(self.attr)
        return ((attr1.view(batch,-1)-attr2.view(batch,-1))**2).sum(-1) # B



class GeomDistance(object):
    def __init__(self, opt):
        geom_opt = opt[('geom',{},"settings for the optimal transport loss")]
        self.attr = geom_opt[('attr','points',"compute distance on the specific class attribute: 'ponts','landmarks','pointfea")]
        args = geom_opt[('args',{'blur':0.1, 'scaling':0.5, 'debais':True},"blur argument in ot")]

        self.gemoloss = SamplesLoss(**args)

    def __call__(self,moved, target):
        attr1 = moved.getattr(self.attr)
        attr2 = target.getattr(self.attr)
        weight1 = moved.weights
        weight2 = target.weights
        return self.gemoloss(weight1,attr1,weight2,attr2)





class Loss():
    """
     DynamicComposed loss class that compose of different loss and has dynamic loss weights during the training
    """

    def __init__(self, opt):
        from shapmagn.global_variable import LOSS_POOL
        loss_opt = opt[('loss', {}, "settings for general loss")]
        loss_name_list = loss_opt[("loss_list",['geomloss'], "a list of loss name to compute: l2, gemoloss, current, varifold")]
        self.loss_weight_strategy_list = loss_opt[("loss_weight_strategy_list",{'l2':'const'}, "for each loss in name_list, design weighting strategy: '{'loss_name':'strategy_param'}")]
        self.loss_activate_epoch_list = loss_opt[("loss_activate_epoch_list",[0], "for each loss in name_list, activate at # epoch'")]
        self.loss_fn_list = [LOSS_POOL[name][loss_opt] for name in loss_name_list ]

    def update_weight(self, epoch):
        if len(self.loss_weight_strategy_list)==0:
            return [1.]* len(self.loss_fn_list)

    def __call__(self, moved, target, epoch):
        weights_list = self.update_weight(epoch)
        loss_list = [weight*loss_fn(moved, target) for weight, loss_fn in zip(weights_list, self.loss_fn_list)]
        total_loss = sum(loss_list)
        return total_loss





# class KernelNorm(object):
#     def __init__(self, opt):
#         kernel_backend = opt[("kernel_backend",'torch',"kernel backend can either be 'torch'/'keops'")]
#         kernel_norm_opt = opt[('kernel_norm',{},"settings for the kernel norm loss")]
#         sigma = kernel_norm_opt[('sigma',0.1,"the sigma in gaussian kernel")]
#         self.kernel = LazyKeopsKernel('gauss',sigma) if kernel_backend == 'keops' else TorchKernel('gauss',sigma)
#     def __call__(self,moved, target):
#         batch = moved.batch
#         c1, n1 = moved.get_centers_and_normals()
#         c2, n2 = target.get_centers_and_normals()
#         def kernel_norm_scalar_product(points_1, points_2, normals_1, normals_2):
#             return ((normals_1.view(batch,-1))
#                     * (self.kernel(points_1, points_2, normals_2))
#                     ).sum(-1)
#
#         return kernel_norm_scalar_product(c1, c1, n1, n1)\
#                + kernel_norm_scalar_product(c2, c2, n2, n2)\
#                - 2 * kernel_norm_scalar_product(c1, c2, n1, n2)
