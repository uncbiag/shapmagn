from torch.autograd import grad
import torch.nn as nn
from shapmagn.utils.obj_factory import obj_factory




############## Hamiltonian view of LDDMM ######################3


class LDDMMHamilton(nn.Module):
    def __init__(self, opt):
        super(LDDMMHamilton,self).__init__()
        self.opt = opt
        kernel = opt[("kernel","torch_kernel.TorchKernel('gauss',0.1)","kernel object")]
        self.kernel = obj_factory(kernel)
        self.mode = "shooting"

    def hamiltonian(self,mom, control_points):
        # todo check, the omitted 1/2 is consistant with variational version
        return (mom * self.kernel(control_points, control_points, mom)).sum()
    def hamiltonian_evolve(self,mom, control_points):
        grad_mom, grad_control = grad(self.hamiltonian(mom, control_points), (mom, control_points), create_graph=True)
        return -grad_control, grad_mom

    def flow(self, mom, control_points, flow_points):
        return (self.kernel(flow_points,control_points,mom),) + self.hamiltonian_evolve(mom, control_points)

    def set_mode(self, mode):
        assert mode in ['shooting','flow']
        self.mode = mode

    def forward(self, t, input):
        if self.mode == "shooting":
            return self.hamiltonian_evolve(*input)
        else:
            return self.flow(*input)


###################  variational view of LDDMM ######################


class LDDMMVariational(nn.Module):
    def __init__(self, opt):
        super(LDDMMVariational, self).__init__()
        self.opt = opt
        kernel = opt[("kernel", "torch_kernel.TorchKernel('gauss',0.1)", "kernel object")]
        self.kernel = obj_factory(kernel)
        grad_kernel = opt[("grad_kernel", "torch_kernel.TorchKernel('gauss_grad',0.1)", "kernel object")]
        self.grad_kernel = obj_factory(grad_kernel)
        self.mode = "shooting"

    def variational_evolve(self,mom, control_points):
        return -self.grad_kernel(mom, control_points), self.kernel(control_points, control_points, mom)

    def variational_flow(self, mom, control_points, flow_points):
        return (self.kernel(flow_points,control_points,mom),) + self.variational_evolve(mom, control_points)


    def set_mode(self, mode):
        assert mode in ['shooting','flow']
        self.mode = mode


    def forward(self,t, input):
        if self.mode == "shooting":
            return self.variational_evolve(*input)
        else:
            return self.variational_flow(*input)

