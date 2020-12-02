import torch
from shapmagn.shape.shape_pair import ShapePair
from shapmagn.global_variable import Shape
from shapmagn.shape.point_interpolator import kernel_interpolator,spline_intepolator


def reg_param_initializer():
    def init(input_data):
        reg_param = torch.zeros_like(input_data["source"]["points"])
        return reg_param
    return init

def create_shape_pair_from_dict(input_data):
    source_dict, target_dict = input_data["source"], input_data["target"]
    shape_pair = ShapePair(dense_mode=True)
    source_shape = Shape()
    source_shape.set_data(points= source_dict['points'],  pointfea=source_dict['point_fea'])
    target_shape = Shape()
    target_shape.set_data(points= target_shape['points'],  pointfea=target_shape['point_fea'])
    shape_pair.set_source_and_target(source_shape, target_shape)
    return shape_pair

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
    reg_param = torch.zeros_like(shape_pair.get_control_points()).normal_(0,1e-7)
    reg_param.requires_grad_()
    shape_pair.set_reg_param(reg_param)
    return shape_pair


def shape_pair_reg_param_lddmm_interpolator(**args):
    model = args['model']
    lddmm_kernel = model.lddmm_kernel
    def interp(shape_pair_low, shape_pair_high):
        points = shape_pair_high.get_control_points()
        control_points = shape_pair_low.get_control_points()
        reg_param_low = shape_pair_low.reg_param
        reg_param_high = lddmm_kernel(points,control_points,reg_param_low)
        reg_param_high.detach_()
        reg_param_high.requires_grad_()
        shape_pair_high.set_reg_param(reg_param_high)
        return shape_pair_high
    return interp


def shape_pair_reg_param_weighted_interpolator(interp_type, **args):
    assert interp_type in ["kernel_interp", "spline_interp"]
    interp_instance = kernel_interpolator(**args) \
        if interp_type=="kernel_interp" else spline_intepolator(**args)

    def interp(shape_pair_low, shape_pair_high):
        points = shape_pair_high.get_control_points()
        control_points = shape_pair_low.get_control_points()
        reg_param_low = shape_pair_low.reg_param
        control_weights = shape_pair_low.control_weights
        reg_param_high = interp_instance(points, control_points,reg_param_low, control_weights)
        reg_param_high.detach_()
        reg_param_high.requires_grad_()
        shape_pair_high.set_reg_param(reg_param_high)
        return shape_pair_high
    return interp




def updater_for_shape_pair_from_low_scale(**args):
    return shape_pair_reg_param_lddmm_interpolator(**args)




