import torch
import torch.nn as nn
from shapmagn.modules.lddmm_module import LDDMMHamilton, LDDMMVariational
from shapmagn.global_variable import Shape
from shapmagn.metrics.losses import Loss
from shapmagn.modules.ode_int import ODEBlock
from shapmagn.utils.utils import sigmoid_decay
class LDDMMOPT(nn.Module):
    def __init__(self, opt):
        super(LDDMMOPT, self).__init__()
        self.opt = opt
        self.module_type = self.opt[("module","hamiltonian", "lddmm module type: hamiltonian or variational")]
        assert self.module_type in ["hamiltonian", "variational"]
        self.lddmm_module = LDDMMHamilton(self.opt[("hamiltonian",{},"settings for hamiltonian")])\
            if self.module_type=='hamiltonian' else LDDMMVariational(self.opt[("variational",{},"settings for variational")])
        self.lddmm_kernel = self.lddmm_module.kernel
        self.interp_kernel = self.lddmm_kernel
        sim_loss_opt = opt[("sim_loss", {}, "settings for sim_loss_opt")]
        self.sim_loss_fn = Loss(sim_loss_opt)
        self.reg_loss_fn = self.geodesic_distance
        self.integrator_opt = self.opt[("integrator", {}, "settings for integrator")]
        self.integrator = ODEBlock(self.integrator_opt)
        self.integrator.set_func(self.lddmm_module)
        self.call_thirdparty_package = False
        self.register_buffer("iter", torch.Tensor([0]))
        self.print_step = self.opt[('print_step',10,"print every n iteration")]



    def init_reg_param(self,shape_pair):
        reg_param = torch.zeros_like(shape_pair.get_control_points()).normal_(0, 1e-7)
        reg_param.requires_grad_()
        shape_pair.set_reg_param(reg_param)
        return shape_pair



    def set_loss_fn(self, loss_fn):
        self.sim_loss_fn = loss_fn

    def reset(self):
        self.iter = self.iter*0




    def shooting(self, shape_pair):
        momentum = shape_pair.reg_param
        control_points = shape_pair.get_control_points()
        self.lddmm_module.set_mode("shooting")
        _, flowed_control_points = self.integrator.solve((momentum, control_points))
        shape_pair.set_flowed_control_points(flowed_control_points)
        return shape_pair

    def flow(self, shape_pair):
        momentum = shape_pair.reg_param
        control_points = shape_pair.control_points
        toflow_points = shape_pair.get_toflow_points()
        self.lddmm_module.set_mode("flow")
        _, flowed_points = self.integrator.solve((momentum, control_points,toflow_points))
        flowed = Shape()
        flowed.set_data_with_refer_to(flowed_points,shape_pair.source)
        shape_pair.set_flowed(flowed)
        return shape_pair

    def geodesic_distance(self,momentum, control_points):
        dist = momentum * self.lddmm_kernel(control_points, control_points, momentum)
        dist = dist.mean()
        return dist

    def get_factor(self):
        """
        get the regularizer factor according to training strategy

        :return:
        """
        sim_factor = 10
        reg_factor_init =1 #self.initial_reg_factor
        static_epoch = 100
        min_threshold = reg_factor_init/10
        decay_factor = 8
        reg_factor = float(
            max(sigmoid_decay(self.iter.item(), static=static_epoch, k=decay_factor) * reg_factor_init, min_threshold))
        return sim_factor, reg_factor


    def update_reg_param_from_low_scale_to_high_scale(self, shape_pair_low, shape_pair_high):
        control_points_high = shape_pair_high.get_control_points()
        control_points_low = shape_pair_low.get_control_points()
        reg_param_low = shape_pair_low.reg_param
        reg_param_high = self.interp_kernel(control_points_high, control_points_low, reg_param_low)
        reg_param_high.detach_()
        reg_param_high.requires_grad_()
        shape_pair_high.set_reg_param(reg_param_high)
        return shape_pair_high


    def forward(self, shape_pair):
        shape_pair = self.shooting(shape_pair)
        flowed_has_inferred = shape_pair.infer_flowed()
        shape_pair = self.flow(shape_pair) if not flowed_has_inferred else shape_pair
        sim_loss = self.sim_loss_fn(shape_pair.flowed, shape_pair.target)
        reg_loss = self.reg_loss_fn(shape_pair.reg_param, shape_pair.get_control_points())
        sim_factor, reg_factor = self.get_factor()
        sim_loss = sim_loss*sim_factor
        reg_loss = reg_loss*reg_factor
        if self.iter%10==0:
            print("{} th step, sim_loss is {}, reg_loss is {}, sim_factor is {}, reg_factor is {}"
                  .format(self.iter.item(), sim_loss.item(), reg_loss.item(),sim_factor, reg_factor))
        loss = sim_loss + reg_loss
        self.iter +=1
        return loss








