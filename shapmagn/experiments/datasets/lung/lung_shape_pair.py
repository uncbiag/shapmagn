
import torch
from shapmagn.shape.shape_pair import ShapePair


def reg_param_initializer():
    def init(input_data):
        reg_param = torch.zeros_like(input_data["source"]["points"])
        return reg_param
    return init





def create_shape_pair(source, target, toflow=None):
    shape_pair = ShapePair().set_source_and_target(source, target)
    if "toflow" is not None:
        shape_pair.set_to_flow(toflow)
    reg_param = torch.zeros_like(shape_pair.get_control_points(), requires_grad=True)
    shape_pair.set_reg_param(reg_param)
    return shape_pair