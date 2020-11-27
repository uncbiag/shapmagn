import torch
import torch.nn as nn
from shapmagn.modules.lddmm_module import LDDMMHamilton, LDDMMVariational
from shapmagn.modules.ode_int import ODEBlock
from shapmagn.global_variable import Shape

class LDDMMOPT(nn.Module):
    def __init__(self, opt):
        super(LDDMMOPT).__init__()
        self.opt = opt[("lddmm_opt",{},"settings for LDDMM optimization")]
        self.module_type = self.opt[("module","hamiltonian", "lddmm module type: hamiltonian or variational")]
        assert self.module_type in ["hamiltonian", "variational"]
        self.module = LDDMMHamilton(self.opt[("hamiltonian",{},"settings for hamiltonian")])\
            if self.module_type=='hamiltonian' else LDDMMVariational(self.opt[("variational",{},"settings for variational")])
        self.integrator_opt = self.opt[("integrator_opt",{},"settings for integrator")]
        self.integrator = ODEBlock(self.integrator_opt)
        self.integrator.set_func(self.module)
        self.sim_loss_fn = None


    def set_loss_fn(self, loss_fn):
        self.sim_loss_fn = loss_fn

    def update_shape_pair(self,shape_pair):
        reg_param = torch.zeros_like(shape_pair.get_control_points(), requires_grad=True)
        shape_pair.set_reg_param(reg_param)
        return shape_pair

    def shooting(self, shape_pair):
        momentum = shape_pair.reg_param
        control_points = shape_pair.control_points
        self.module.set_mode("shooting")
        _, flowed_control_points = self.integrator.solve((momentum, control_points))
        shape_pair.set_flowed_control_points(flowed_control_points)
        return shape_pair

    def flow(self, shape_pair):
        momentum = shape_pair.reg_param
        control_points = shape_pair.control_points
        toflow_points = shape_pair.get_toflow_points()
        self.module.set_mode("flow")
        _, flowed_points = self.integrator.solve((momentum, control_points,toflow_points))
        flowed = Shape().set_data_with_refer_to(flowed_points,shape_pair.source)
        shape_pair.set_flowed(flowed)
        return shape_pair

    def geodesic_distance(self,momentum, control_points):
        dist = momentum * self.module.kernel(control_points, control_points, momentum)
        dist = dist.sum()
        return dist


    def forward(self, shape_pair):
        shape_pair = self.update_shape_pair(shape_pair)
        shape_pair = self.shooting(shape_pair)
        flowed_has_infered = shape_pair.infer_flowed()
        shape_pair = self.flow(shape_pair) if not flowed_has_infered else shape_pair
        sim_loss = self.sim_loss_fn(shape_pair.flowed, shape_pair.target)
        reg_loss = self.geodesic_distance(shape_pair.reg_param, shape_pair.get_control_points())
        loss = sim_loss + reg_loss
        return loss








