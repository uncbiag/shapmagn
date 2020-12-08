import torch
import torch.nn as nn
from shapmagn.global_variable import Shape
from shapmagn.metrics.losses import Loss
from shapmagn.utils.obj_factory import obj_factory
from shapmagn.utils.utils import sigmoid_decay
from torch.autograd import grad
class BayCenterOPT(nn.Module):
    def __init__(self, opt):
        super(BayCenterOPT, self).__init__()
        self.opt = opt
        interpolator_obj = self.opt[("interpolator_obj","kernel_interpolator(scale=0.1, exp_order=2)", "shape interpolator")]
        self.interp_kernel = obj_factory(interpolator_obj)
        sim_loss_opt = opt[("sim_loss", {}, "settings for sim_loss_opt")]
        self.sim_loss_fn = Loss(sim_loss_opt)
        self.reg_loss_fn = self.geodesic_distance
        self.register_buffer("iter", torch.Tensor([0]))
        self.print_step = self.opt[('print_step',10,"print every n iteration")]




    def set_loss_fn(self, loss_fn):
        self.sim_loss_fn = loss_fn

    def reset_iter(self):
        self.iter = self.iter*0



    def flow(self, shape_pair):
        control_flow = shape_pair.reg_param
        toflow_points = shape_pair.toflow.points
        control_points = shape_pair.control_points
        control_weights = shape_pair.control_weights
        flow = self.interp_kernel(toflow_points,control_points, control_flow,control_weights)
        flowed_points =  toflow_points + flow
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


    def forward(self, shape_pair):
        """
        flowed(t) = control(t-1) + flow(t-1)
        grad_flow(t) = Gradient(loss(flowed(t),target))
        flow(t) = - grad_flow(t)/weight
        control(t) = flowed(t)
        :param shape_pair:
        :return:
        """
        flowed_control_points = shape_pair.control_points + shape_pair.reg_param
        flowed_control_points.detach_()
        flowed_control_points.requires_grad_()
        shape_pair.set_flowed_control_points(flowed_control_points)
        flowed_has_inferred = shape_pair.infer_flowed()
        shape_pair = self.flow(shape_pair) if not flowed_has_inferred else shape_pair
        sim_loss = self.sim_loss_fn(shape_pair.flowed, shape_pair.target)
        reg_loss = self.reg_loss_fn(shape_pair.reg_param, shape_pair.get_flowed_points())
        sim_factor, reg_factor = self.get_factor()
        sim_loss = sim_loss*sim_factor
        reg_loss = reg_loss*reg_factor
        if self.iter%10==0:
            print("{} th step, sim_loss is {}, reg_loss is {}, sim_factor is {}, reg_factor is {}"
                  .format(self.iter.item(), sim_loss.item(), reg_loss.item(),sim_factor, reg_factor))
        loss = sim_loss + reg_loss
        grad_flow = grad(loss,shape_pair.flowed_control_points)
        shape_pair.set_control_points(shape_pair.control_points.detach() + shape_pair.reg_param.detach())
        shape_pair.reg_param = - grad_flow/shape_pair.control_weights
        self.iter +=1
        return loss








