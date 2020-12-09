import torch
import torch.nn as nn
from shapmagn.global_variable import Shape
from shapmagn.metrics.losses import Loss
from shapmagn.utils.obj_factory import obj_factory
from shapmagn.utils.utils import sigmoid_decay
from torch.autograd import grad
class GradientFlowOPT(nn.Module):
    def __init__(self, opt):
        super(GradientFlowOPT, self).__init__()
        self.opt = opt
        interpolator_obj = self.opt[("interpolator_obj","kernel_interpolator(scale=0.1, exp_order=2)", "shape interpolator")]
        gauss_kernel_obbj = opt[("gauss_kernel_obj","torch_kernels.TorchKernel('gauss',sigma=0.1)","kernel object")]
        self.gauss_kernel = obj_factory(gauss_kernel_obbj)
        self.interp_kernel = obj_factory(interpolator_obj)
        sim_loss_opt = opt[("sim_loss", {}, "settings for sim_loss_opt")]
        self.sim_loss_fn = Loss(sim_loss_opt)
        self.reg_loss_fn = self.geodesic_distance
        self.register_buffer("iter", torch.Tensor([0]))
        self.print_step = self.opt[('print_step',1,"print every n iteration")]


    def set_loss_fn(self, loss_fn):
        self.sim_loss_fn = loss_fn

    def reset(self,shape_pair, control_points):
        self.iter = self.iter*0
        shape_pair.set_control_points(control_points)
        return shape_pair



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

    def geodesic_distance(self,flow, control_points):
        dist = flow * self.gauss_kernel(control_points, control_points, flow)
        dist = dist.mean()
        return dist



    def forward(self, shape_pair):
        """
        flowed(t) = control(t-1) + flow(t-1)
        grad_flow(t) = Gradient(sim_loss(flowed(t),target)+reg_loss(flow(t-1),control(t-1)))
        flow(t) = - grad_flow(t)/weight
        control(t) = flowed(t)
        :param shape_pair:
        :return:
        """
        shape_pair.reg_param.requires_grad = False
        flowed_control_points = shape_pair.get_control_points() + shape_pair.reg_param
        flowed_control_points.detach_()
        flowed_control_points.requires_grad_()
        shape_pair.set_flowed_control_points(flowed_control_points)
        flowed_has_inferred = shape_pair.infer_flowed()
        shape_pair = self.flow(shape_pair) if not flowed_has_inferred else shape_pair
        sim_loss = self.sim_loss_fn(shape_pair.flowed, shape_pair.target)
        sim_loss = sim_loss
        if self.iter%self.print_step==0:
            print("{} th step, sim_loss is {}".format(self.iter.item(), sim_loss.item(),))
        loss = sim_loss
        grad_flow = grad(loss,shape_pair.flowed_control_points)[0]
        shape_pair.set_control_points(shape_pair.control_points.detach() + shape_pair.reg_param.detach())
        shape_pair.reg_param = - grad_flow/shape_pair.control_weights
        self.iter +=1
        return loss








