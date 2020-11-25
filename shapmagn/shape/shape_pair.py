"""
a ShapePair object records the registration parameters of the source shape to the target shape
"""
import torch

class ShapePair():
    def __init__(self):
        self.source = None
        self.target = None
        self.moved = None
        self.reg_param = None
        self.control_points = None

    def set_source_and_target(self, source, target):
        self.source = source
        self.moved = source.clone()
        self.target = target
        self.reg_param = torch.zeros_like(source.points)

    def set_init_control_points(self, control_points):
        self.control_points = control_points
        self.reg_param = torch.zeros_like(control_points)

    def get_init_control_points(self):
        if self.control_points is not None:
            return self.control_points
        else:
            return self.moved.points
