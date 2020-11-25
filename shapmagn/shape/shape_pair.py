"""
a ShapePair object records the registration parameters of the source shape to the target shape
"""
import torch
from shapmagn.global_variable import Shape

class ShapePair():
    """
    1. During training, the control points are flowed to get flowed control points.
    if the dense mode is true, which refers the control points is the same as the source,
    the flowed then can be directly created from  the flowed control points
    if the dense mode is false, which refers the control points are different from the source,
    , the toflow is set as the same as the source, additional inference on toflow is needed to get
    the flowed
    The similarity measure will be computed between the flowed and the target

    2. During external inference, e.g. given ambient points, in this case, assume the reg_param
    and the control points have already known. the dense mode is set to false, the  toflow need to be
    initialized externally. The flowed can be return after the inference.
    """
    def __init__(self):
        self.source = None
        self.target = None
        self.toflow = None
        self.flowed = None
        self.reg_param = None
        self.control_points = None
        self.flowed_control_points = None
        self.dense_mode = True

    def set_source_and_target(self, source, target):
        self.source = source
        self.target = target
        self.toflow = source.clone()

    def set_to_flow(self, toflow):
        self.toflow = toflow
        self.dense_mode = False

    def set_flowed(self, flowed):
        self.flowed = flowed

    def set_reg_param(self, reg_param):
        self.reg_param = reg_param

    def set_flowed_control_points(self, flowed_control_points, dense_mode):
        self.flowed_control_points = flowed_control_points
        self.dense_mode = dense_mode

    def infer_flowed(self):
        if self.dense_mode:
            self.flow = Shape().set_data_with_refer_to(self.flowed_control_points,self.toflow)
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
