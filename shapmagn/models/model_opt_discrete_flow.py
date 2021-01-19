import torch
import torch.nn as nn
from shapmagn.global_variable import Shape
from shapmagn.metrics.losses import Loss
from shapmagn.utils.obj_factory import obj_factory
from shapmagn.utils.utils import sigmoid_decay
from shapmagn.modules.optimizer import optimizer_builder
from shapmagn.modules.scheduler import scheduler_builder
from torch.autograd import grad
class DiscreteFlowOPT(nn.Module):
    def __init__(self, opt):
        super(DiscreteFlowOPT, self).__init__()
        self.opt = opt
        interpolator_obj = self.opt[("interpolator_obj","point_interpolator.kernel_interpolator(scale=0.1, exp_order=2)", "shape interpolator")]
        gauss_kernel_obj = opt[("gauss_kernel_obj","torch_kernels.TorchKernel('gauss',sigma=0.1)","kernel object")]
        self.gauss_kernel = obj_factory(gauss_kernel_obj)
        self.interp_kernel = obj_factory(interpolator_obj)
        sim_loss_opt = opt[("sim_loss", {}, "settings for sim_loss_opt")]
        self.sim_loss_fn = Loss(sim_loss_opt)
        self.reg_loss_fn = self.geodesic_distance
        self.call_thirdparty_package = False
        self.register_buffer("iter", torch.Tensor([0]))
        self.print_step = self.opt[('print_step',1,"print every n iteration")]
        self.opt_optim = opt['optim']
        """settings for the optimizer"""
        self.opt_scheduler = opt['scheduler']
        """settings for the scheduler"""



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

    def geodesic_distance(self,flow, control_points):
        dist = flow * self.gauss_kernel(control_points, control_points, flow)
        dist = dist.mean()
        return dist




    def update_reg_param_from_low_scale_to_high_scale(self, shape_pair_low, shape_pair_high):
        control_points_high = shape_pair_high.get_control_points()
        control_points_low = shape_pair_low.get_control_points()
        control_weights_low = shape_pair_low.control_weights
        reg_param_low = shape_pair_low.reg_param
        reg_param_high = self.interp_kernel(control_points_high, control_points_low, reg_param_low, control_weights_low)
        flowed_control_points_low = shape_pair_low.flowed_control_points.detach()
        interped_control_points_high = self.interp_kernel(control_points_high, control_points_low,
                                                     flowed_control_points_low, control_weights_low)
        shape_pair_high.set_control_points(interped_control_points_high)
        reg_param_high.detach_()
        reg_param_high.requires_grad_()
        shape_pair_high.set_reg_param(reg_param_high)
        return shape_pair_high




    def init_reg_param(self,shape_pair):
        reg_param = torch.zeros_like(shape_pair.get_control_points()).normal_(0, 1e-7)
        reg_param.requires_grad_()
        shape_pair.set_reg_param(reg_param)
        return shape_pair


    def extract_point_fea(self, flowed, target):
        flowed.pointfea = flowed.points
        target.pointfea = target.points
        return flowed, target

    def extract_fea(self, flowed, target):
        """DiscreteFlowOPT supports feature extraction"""
        if not self.feature_extractor:
            return self.extract_point_fea(flowed, target)
        else:
            raise ValueError("the feature extraction approach {} hasn't implemented".format(self.feature_extractor))

    def forward(self, shape_pair):
        """
        reg_param here is the displacement
        :param shape_pair:
        :return:
        """
        flowed_control_points = shape_pair.get_control_points() + shape_pair.reg_param
        shape_pair.set_flowed_control_points(flowed_control_points)
        flowed_has_inferred = shape_pair.infer_flowed()
        shape_pair = self.flow(shape_pair) if not flowed_has_inferred else shape_pair
        shape_pair.flowed, shape_pair.target = self.extract_fea(shape_pair.flowed, shape_pair.target)
        sim_loss = self.sim_loss_fn(shape_pair.flowed, shape_pair.target)
        reg_loss = self.reg_loss_fn(shape_pair.reg_param,shape_pair.get_control_points())
        sim_factor, reg_factor = self.get_factor()
        sim_loss = sim_loss * sim_factor
        reg_loss = reg_loss * reg_factor
        if self.iter % 10 == 0:
            print("{} th step, sim_loss is {}, reg_loss is {}, sim_factor is {}, reg_factor is {}")
        loss = sim_loss + reg_loss
        # shape_pair.set_control_points(shape_pair.control_points.detach() + shape_pair.reg_param.detach())
        # shape_pair.reg_param.detach_()
        # shape_pair.set_flowed_control_points(shape_pair.reg_param)
        # shape_pair.infer_flowed()
        # shape_pair.reg_param.requires_grad = True
        self.iter += 1
        return loss






