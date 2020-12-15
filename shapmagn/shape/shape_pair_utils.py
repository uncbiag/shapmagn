import torch
from shapmagn.shape.shape_pair import ShapePair
from shapmagn.global_variable import Shape
from shapmagn.shape.point_interpolator import kernel_interpolator,spline_intepolator


def reg_param_initializer():
    def init(input_data):
        reg_param = torch.zeros_like(input_data["source"]["points"])
        return reg_param
    return init


def create_source_and_target_shape():
    def create(input_dict):
        source_dict, target_dict = input_dict["source"], input_dict["target"]
        source_shape = Shape()
        source_shape.set_data(**source_dict)
        target_shape = Shape()
        target_shape.set_data(**target_dict)
        return source_shape, target_shape
    return create

def create_shape_pair(source, target, toflow=None):
    shape_pair = ShapePair()
    shape_pair.set_source_and_target(source, target)
    if toflow is not None:
        shape_pair.set_toflow(toflow)
    return shape_pair





