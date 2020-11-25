import torch
import torch.nn as nn
from shapmagn.modules.lddmm_module import LDDMMHamilton, LDDMMVariational
from shapmagn.modules.ode_int import ODEBlock
from shapmagn.global_variable import Shape
from shapmagn.shape.shape_pair import ShapePair

class LDDMMOPT(nn.Module):
    def __init__(self, opt):
        super(LDDMMOPT).__init__()
        self.opt = opt[("lddmm_opt",{},"settings for LDDMM optimization")]
        self.module_type = self.opt[("module","hamiltonian", "lddmm module type: hamiltonian or variational")]
        assert self.module_type in ["hamiltonian", "variational"]
        self.module = LDDMMHamilton if self.module_type=='hamiltonian' else LDDMMVariational
        self.integrator_opt = self.opt[("integrator_opt",{},"settings for integrator")]
        self.integrator = ODEBlock(self.integrator_opt)
        self.integrator.set_func(self.module)
        self.sim_loss_fn = None




    def set_loss_fn(self, loss_fn):
        self.sim_loss_fn = loss_fn

    def update_shape_pair(self,shape_pair):
        reg_param = torch.zeros_like(shape_pair.source.points)
        shape_pair.set_reg_param(reg_param)
        return shape_pair

    def shooting(self, shape_pair):
        momentum = shape_pair.reg_param
        control_points = shape_pair.control_points
        _, moved_points = self.integrator.solve((momentum, control_points))
        moved = Shape().set_data_with_refer_to(moved_points,shape_pair.source)
        shape_pair.set_moved(moved)
        return shape_pair

    def geodesic_distance(self,momentum, position):
        dist = momentum * self.module.kernel(position, position, momentum)
        dist = dist.sum()
        return dist


    def forward(self, shape_pair):
        shape_pair = self.update_shape_pair(shape_pair)
        shape_pair = self.shooting(shape_pair)
        sim_loss = self.sim_loss_fn(shape_pair.moved, shape_pair.target)
        reg_loss = self.geodesic_distance(shape_pair.reg_param, shape_pair.source.points)
        loss = sim_loss + reg_loss
        return loss








