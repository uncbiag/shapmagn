"""
a ShapePair object records the registration parameters of the source shape to the target shape
"""
import torch
from shapmagn.global_variable import Shape

class ShapePair():
    """
    1. During training, the control points are flowed to get flowed control points.

    if the dense mode is true, which refers the control points is the same as the source,
    the flowed then can be directly created from the flowed control points
    (toflow and flowed are implicitly set)

    Examples:
        >>> shape_pair = ShapePair(dense_mode=True)
        >>> shape_pair.set_source_and_target(source, target)
        >>> do_registration(shape_pair)


    if the dense mode is false, which refers the control points are different from the source,
    , the toflow by default is set as the source, additional inference on the toflow is needed to get the flowed
    The similarity measure will be computed between the flowed and the target

    Examples:
        >>> shape_pair = ShapePair(dense_mode=False)
        >>> shape_pair.set_source_and_target(source, target)
        >>> shape_pair.set_control_points(contorl_points)
        >>> do_registration(shape_pair)
        >>> do_flow(shape_pair)


    2. During external inference, e.g. given ambient points, in this case, assume the reg_param
    and the control points have already known. the dense mode is set to false, the  toflow need to be
    initialized externally. The flowed can be return after the inference.

    Examples:
        >>> ....
        >>> shape_pair.set_toflow(toflow)
        >>> do_flow(shape_pair)
    """
    def __init__(self, dense_mode=True):
        self.source = None
        self.target = None
        self.toflow = None
        self.flowed = None
        self.reg_param = None
        self.control_points = None
        self.flowed_control_points = None
        self.dense_mode = dense_mode
        self.extra_info = None

    def set_source_and_target(self, source, target):
        self.source = source
        self.target = target
        self.toflow = source.clone()

    def set_toflow(self, toflow):
        self.toflow = toflow
        self.dense_mode = False

    def set_flowed(self, flowed):
        self.flowed = flowed

    def set_reg_param(self, reg_param):
        self.reg_param = reg_param

    def set_flowed_control_points(self, flowed_control_points):
        self.flowed_control_points = flowed_control_points

    def infer_flowed(self):
        if self.dense_mode:
            self.flowed = Shape().set_data_with_refer_to(self.flowed_control_points,self.toflow)
            return True
        else:
            return False


    def set_control_points(self, control_points):
        self.control_points = control_points

    def get_control_points(self):
        if self.control_points is not None:
            return self.control_points
        else:
            return self.source.points.clone()

    def get_toflow_points(self):
        return self.toflow.points.clone()
