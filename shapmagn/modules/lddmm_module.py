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

    def hamiltonian(self,mom, pos):
        # todo check, the omitted 1/2 is consistant with variational version
        return (mom * self.kernel(pos, pos, mom)).sum()
    def hamiltonian_evolve(self,mom, pos):
        grad_mom, grad_pos = grad(self.hamiltonian(mom, pos), (mom, pos), create_graph=True)
        return -grad_pos, grad_mom

    def forward(self,mom, pos):
        return self.hamiltonian_evolve(mom,pos)



###################  variational view of LDDMM ######################


class LDDMMVariational(nn.Module):
    def __init__(self, opt):
        super(LDDMMVariational, self).__init__()
        self.opt = opt
        kernel = opt[("kernel", "torch_kernel.TorchKernel('gauss',0.1)", "kernel object")]
        self.kernel = obj_factory(kernel)
        grad_kernel = opt[("kernel", "torch_kernel.TorchKernel('gauss_grad',0.1)", "kernel object")]
        self.grad_kernel = obj_factory(grad_kernel)

    def variational_evolve(self,mom, pos):
        return -self.grad_kernel(mom, pos), self.kernel(pos, pos, mom)

    def forward(self, mom, pos):
        return self.variational_evolve(mom, pos)


