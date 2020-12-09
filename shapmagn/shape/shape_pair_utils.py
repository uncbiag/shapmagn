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
    reg_param = torch.zeros_like(shape_pair.get_control_points()).normal_(0,1e-7)
    reg_param.requires_grad_()
    shape_pair.set_reg_param(reg_param)
    return shape_pair


def shape_pair_reg_param_interpolator(**args):
    model = args['model']
    task_type = args['task_type']
    interp_control_points = True if task_type == "gradient_flow" else False
    interp_kernel = model.interp_kernel
    def interp(shape_pair_low, shape_pair_high):
        control_points_high = shape_pair_high.get_control_points()
        control_points_low = shape_pair_low.get_control_points()
        control_weights_low = shape_pair_low.control_weights
        reg_param_low = shape_pair_low.reg_param
        reg_param_high = interp_kernel(control_points_high,control_points_low,reg_param_low,control_weights_low)
        if interp_control_points:
            flowed_control_points_low = shape_pair_low.flowed_control_points.detach()
            interped_control_points_high = interp_kernel(control_points_high,control_points_low,flowed_control_points_low,control_weights_low)
            shape_pair_high.set_control_points(interped_control_points_high)
        reg_param_high.detach_()
        reg_param_high.requires_grad_()
        shape_pair_high.set_reg_param(reg_param_high)
        return shape_pair_high
    return interp




def updater_for_shape_pair_from_low_scale(**args):

    return shape_pair_reg_param_interpolator(**args)




