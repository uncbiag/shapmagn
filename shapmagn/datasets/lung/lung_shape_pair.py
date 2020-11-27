import torch
from shapmagn.shape.shape_pair import ShapePair
from shapmagn.global_variable import Shape


def init_reg_param(input_data):
    reg_param = torch.zeros_like(input_data)
    return reg_param




def init_shape_pair_during_train(input_data):
    source_dict, target_dict = input_data["source"], input_data["target"]
    shape_pair = ShapePair(dense_mode=True)
    source_shape = Shape()
    source_shape.set_data(points= source_dict['points'],  pointfea=source_dict['point_fea'])
    target_shape = Shape()
    target_shape.set_data(points= target_shape['points'],  pointfea=target_shape['point_fea'])
    shape_pair.set_source_and_target(source_shape, target_shape)
    return shape_pair

