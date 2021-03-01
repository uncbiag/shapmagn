import torch
from shapmagn.shape.shape_pair import ShapePair
from shapmagn.global_variable import Shape
from shapmagn.utils.obj_factory import obj_factory
from shapmagn.experiments.datasets.lung.lung_data_aug import *



def reg_param_initializer():
    def init(input_data):
        reg_param = torch.zeros_like(input_data["source"]["points"])
        return reg_param
    return init




def init_shape_pair(input_data):
    source_dict, target_dict = input_data["source"], input_data["target"]
    shape_pair = ShapePair(dense_mode=True)
    source_shape = Shape()
    source_shape.set_data(points= source_dict['points'], weights=source_dict["weights"], pointfea=source_dict['point_fea'])
    target_shape = Shape()
    target_shape.set_data(points= target_dict['points'], weights=target_dict["weights"], pointfea=target_dict['point_fea'])
    shape_pair.set_source_and_target(source_shape, target_shape)
    return shape_pair





def prepare_synth_input():
    synthsizer = lung_synth_data()
    def prepare(input_data, batch_info):
        synth_on_source = random.random() > 0.5
        source_dict = input_data["source"] if synth_on_source else input_data["target"]
        input_data["target"], synth_info = synthsizer(deepcopy(source_dict))
        input_data["source"] = source_dict
        batch_info["source_info"] = batch_info["source_info"] if synth_on_source else batch_info["target_info"]
        batch_info["target_info"] = batch_info["source_info"]
        batch_info["pair_name"] = [name +"_and_synth" for name in batch_info["source_info"]["name"]]
        batch_info["synth_info"] = synth_info
        batch_info["is_synth"] = True
        return input_data, batch_info
    return prepare









def create_shape_pair(source, target, toflow=None):
    shape_pair = ShapePair().set_source_and_target(source, target)
    if "toflow" is not None:
        shape_pair.set_to_flow(toflow)
    reg_param = torch.zeros_like(shape_pair.get_control_points(), requires_grad=True)
    shape_pair.set_reg_param(reg_param)
    return shape_pair