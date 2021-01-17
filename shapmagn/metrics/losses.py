"""
the code is largely borrowed from deformetrica
here turns it into a batch version
"""

import torch
from shapmagn.kernels.keops_kernels import LazyKeopsKernel
from shapmagn.kernels.torch_kernels import TorchKernel
from shapmagn.utils.obj_factory import obj_factory

class CurrentDistance(object):
    def __init__(self, opt):
        kernel_backend = opt[("kernel_backend",'torch',"kernel backend can either be 'torch'/'keops'")]
        sigma = opt[('sigma',0.1,"the sigma in gaussian kernel")]
        self.kernel = LazyKeopsKernel('gauss',sigma=sigma) if kernel_backend == 'keops' else TorchKernel('gauss',sigma=sigma)

    def __call__(self,flowed, target):
        assert flowed.type == 'PolyLine'
        batch = flowed.batch
        c1, n1 = flowed.get_centers_and_currents()
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
        kernel_backend = opt[("kernel_backend", 'torch', "kernel backend can either be 'torch'/'keops'")]
        sigma = opt[('sigma', 0.1, "the sigma in gaussian lin kernel")]
        self.kernel = LazyKeopsKernel('gauss_lin', sigma=sigma) if kernel_backend == 'keops' else TorchKernel('gauss_lin', sigma=sigma)
    def __call__(self, flowed, target):
        assert flowed.type == 'SurfaceMesh'
        batch = flowed.batch
        c1, n1 = flowed.get_centers_and_normals()
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
        self.attr = opt[('attr','landmarks',"compute distance on the specific class attribute: 'ponts','landmarks','pointfea")]
    def __call__(self,flowed, target):
        batch = flowed.nbatch
        attr1 = flowed.getattr(self.attr)
        attr2 = target.getattr(self.attr)
        return ((attr1.view(batch,-1)-attr2.view(batch,-1))**2).sum(-1) # B



class GeomDistance(object):
    def __init__(self, opt):
        self.attr = opt[('attr','points',"compute distance on the specific class attribute: 'ponts','landmarks','pointfea")]
        geom_obj = opt[('geom_obj',{'blur':0.1, 'scaling':0.5, 'debais':True},"blur argument in ot")]

        self.gemoloss = obj_factory(geom_obj)

    def __call__(self,flowed, target):
        attr1 = getattr(flowed,self.attr)
        attr2 = getattr(target,self.attr)
        weight1 = flowed.weights[:,:,0] #remove the last dim
        weight2 = target.weights[:,:,0] #remove the last dim
        return self.gemoloss(weight1,attr1,weight2,attr2)





class Loss():
    """
     DynamicComposed loss class that compose of different loss and has dynamic loss weights during the training
    """

    def __init__(self, opt):
        from shapmagn.global_variable import LOSS_POOL
        loss_name_list = opt[("loss_list",['l2'], "a list of loss name to compute: l2, gemoloss, current, varifold")]
        self.loss_weight_strategy_dict = opt[("loss_weight_strategy","", "for each loss in name_list, design weighting strategy: '{'loss_name':'strategy_param'}")]
        self.loss_activate_epoch_list = opt[("loss_activate_epoch_list",[0], "for each loss in name_list, activate at # epoch'")]
        self.loss_fn_list = [LOSS_POOL[name](opt[(name,{},"settings")]) for name in loss_name_list ]

    def update_weight(self, epoch):
        if len(self.loss_weight_strategy_dict)==0:
            return [1.]* len(self.loss_fn_list)

    def __call__(self, flowed, target, epoch=-1):
        weights_list = self.update_weight(epoch)
        loss_list = [weight*loss_fn(flowed, target) for weight, loss_fn in zip(weights_list, self.loss_fn_list)]
        total_loss = sum(loss_list)
        return total_loss





# class KernelNorm(object):
#     def __init__(self, opt):
#         kernel_backend = opt[("kernel_backend",'torch',"kernel backend can either be 'torch'/'keops'")]
#         kernel_norm_opt = opt[('kernel_norm',{},"settings for the kernel norm loss")]
#         sigma = kernel_norm_opt[('sigma',0.1,"the sigma in gaussian kernel")]
#         self.kernel = LazyKeopsKernel('gauss',sigma) if kernel_backend == 'keops' else TorchKernel('gauss',sigma)
#     def __call__(self,flowed, target):
#         batch = flowed.batch
#         c1, n1 = flowed.get_centers_and_normals()
#         c2, n2 = target.get_centers_and_normals()
#         def kernel_norm_scalar_product(points_1, points_2, normals_1, normals_2):
#             return ((normals_1.view(batch,-1))
#                     * (self.kernel(points_1, points_2, normals_2))
#                     ).sum(-1)
#
#         return kernel_norm_scalar_product(c1, c1, n1, n1)\
#                + kernel_norm_scalar_product(c2, c2, n2, n2)\
#                - 2 * kernel_norm_scalar_product(c1, c2, n1, n2)
