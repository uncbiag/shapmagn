import torch
from shapmagn.shape.shape_pair import ShapePair
from shapmagn.global_variable import Shape, shape_type
from shapmagn.shape.point_interpolator import nadwat_kernel_interpolator,ridge_kernel_intepolator


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

def create_shape_pair(source, target, toflow=None,pair_name=None):
    shape_pair = ShapePair()
    shape_pair.set_source_and_target(source, target)
    if toflow is not None:
        shape_pair.set_toflow(toflow)
    if pair_name is not None:
        shape_pair.set_pair_name(pair_name)
    return shape_pair


def create_shape_pair_from_data_dict():
    def create(data_dict):
        shape_pair = ShapePair()
        source_dict, target_dict = data_dict["source"], data_dict["target"]
        source = Shape()
        source.set_data(**source_dict)
        target = Shape()
        target.set_data(**target_dict)
        shape_pair.set_source_and_target(source, target)
        if "toflow" in data_dict:
            toflow = Shape()
            toflow.set_data(**data_dict["toflow"])
            shape_pair.toflow = toflow
        if "flowed" in data_dict:
            flowed = Shape()
            flowed.set_data(**data_dict["flowed"])
            shape_pair.flowed = flowed
        if "reg_param" in data_dict:
            shape_pair.reg_param = data_dict["reg_param"]
        if "control_points" in data_dict:
            shape_pair.control_points = data_dict["control_points"]
        if "control_weights" in data_dict:
            shape_pair.control_weights = data_dict["control_weights"]
        if "flowed_control_points" in data_dict:
            shape_pair.flowed_control_points = data_dict["flowed_control_points"]
        if "extra_info" in data_dict:
            shape_pair.extra_info = data_dict["extra_info"]
        return shape_pair
    return create



def decompose_shape_into_dict(shape):
    data_dict = {attr:getattr(shape,attr) for attr in shape.attr_list if getattr(shape,attr) is not None}
    return data_dict

def decompose_shape_pair_into_dict():
    def decompose(shape_pair):
        data_dict = {}
        data_dict["source"] = decompose_shape_into_dict(shape_pair.source)
        data_dict["target"] = decompose_shape_into_dict(shape_pair.target)
        if shape_pair.toflow is not None:
            data_dict["toflow"] = decompose_shape_into_dict(shape_pair.toflow)
        if shape_pair.flowed is not None:
            data_dict["flowed"] = decompose_shape_into_dict(shape_pair.flowed)
        if shape_pair.reg_param is not None:
            data_dict["reg_param"] = shape_pair.reg_param
        if shape_pair.control_points is not None:
            data_dict["control_points"] = shape_pair.control_points
        if shape_pair.control_weights is not None:
            data_dict["control_weights"] = shape_pair.control_weights
        if shape_pair.flowed_control_points is not None:
            data_dict["flowed_control_points"] = shape_pair.flowed_control_points
        if shape_pair.extra_info is not None:
            data_dict["extra_info"] = shape_pair.extra_info
        return data_dict
    return decompose









