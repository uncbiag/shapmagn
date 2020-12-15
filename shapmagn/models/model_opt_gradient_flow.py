import torch
import torch.nn as nn
from shapmagn.global_variable import Shape
from shapmagn.metrics.losses import Loss
from shapmagn.utils.obj_factory import obj_factory
from torch.autograd import grad
class GradientFlowOPT(nn.Module):
    def __init__(self, opt):
        super(GradientFlowOPT, self).__init__()
        self.opt = opt
        interpolator_obj = self.opt[("interpolator_obj","point_interpolator.kernel_interpolator(scale=0.1, exp_order=2)", "shape interpolator")]
        self.interp_kernel = obj_factory(interpolator_obj)
        sim_loss_opt = opt[("sim_loss", {}, "settings for sim_loss_opt")]
        self.sim_loss_fn = Loss(sim_loss_opt)
        self.call_thirdparty_package = False
        self.register_buffer("iter", torch.Tensor([0]))
        self.print_step = self.opt[('print_step',1,"print every n iteration")]


    def set_loss_fn(self, loss_fn):
        self.sim_loss_fn = loss_fn

    def reset(self):
        self.iter = self.iter*0



    def flow(self, shape_pair):
        flowed_control_points = shape_pair.flowed_control_points
        toflow_points = shape_pair.toflow.points
        control_points = shape_pair.control_points
        control_weights = shape_pair.control_weights
        flowed_points = self.interp_kernel(toflow_points,control_points, flowed_control_points,control_weights)
        flowed = Shape()
        flowed.set_data_with_refer_to(flowed_points,shape_pair.source)
        shape_pair.set_flowed(flowed)
        return shape_pair




    def update_reg_param_from_low_scale_to_high_scale(self, shape_pair_low, shape_pair_high):
        control_points_high = shape_pair_high.get_control_points()
        control_points_low = shape_pair_low.get_control_points()
        control_weights_low = shape_pair_low.control_weights
        reg_param_low = shape_pair_low.reg_param
        reg_param_high = self.interp_kernel(control_points_high, control_points_low, reg_param_low, control_weights_low)
        reg_param_high.detach_()
        reg_param_high.requires_grad_()
        shape_pair_high.set_reg_param(reg_param_high)
        return shape_pair_high



    def init_reg_param(self,shape_pair):
        reg_param = shape_pair.get_control_points().clone().detach()
        reg_param.requires_grad_()
        shape_pair.set_reg_param(reg_param)
        return shape_pair

    def forward(self, shape_pair):
        """
        reg_param here is the displacement
        flowed_control(t) = reg_param(t)
        flowed(t) = interpolate(flowed_control(t))
        grad_on_reg_param(t) = Gradient(sim_loss(flowed(t),target)
        reg_param(t) += -(grad_on_reg_param)/control_weights
        :param shape_pair:
        :return:
        """
        flowed_control_points = shape_pair.reg_param
        shape_pair.set_flowed_control_points(flowed_control_points)
        flowed_has_inferred = shape_pair.infer_flowed()
        shape_pair = self.flow(shape_pair) if not flowed_has_inferred else shape_pair
        sim_loss = self.sim_loss_fn(shape_pair.flowed, shape_pair.target)
        loss = sim_loss
        print("{} th step, sim_loss is {}".format(self.iter.item(), sim_loss.item(),))
        grad_reg_param = grad(loss,shape_pair.shape_pair.reg_param)[0]
        shape_pair.reg_param += - grad_reg_param/shape_pair.control_weights
        shape_pair.reg_param.detach_()
        shape_pair.set_flowed_control_points(shape_pair.reg_param)
        shape_pair.infer_flowed()
        shape_pair.reg_param.requires_grad = True
        self.iter +=1
        return loss.detach()






