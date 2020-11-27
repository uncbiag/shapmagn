import torch
import torch.nn as nn
import torch.optim as optim


w1 = torch.randn(3, 3)
w1.requires_grad = True
w2 = torch.randn(3, 3)
w2.requires_grad = True
class MyModule(nn.Module):
    def __init__(self):
        # you need to register the parameter names earlier
        super(MyModule).__init__()

    def reset_parameters(self, input):
        self.weight = nn.Parameter(input.new(input.size()).normal_(0, 1))

    def forward(self, input):
        self.register_parameter('weight', None)

        if self.weight is None:
            self.reset_parameters(input)
            return self.weight @ input
aa = MyModule()
opt_instance = optim.SGD(aa.parameters(),lr=1)