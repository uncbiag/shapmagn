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



class HybirdData(object):
    def __init__(self, synthsizer_obj, data_aug_obj=None, synth_ratio=1.0, ratio_decay_rate=8, min_synth_ratio=0.3):
        super(HybirdData,self).__init__()
        self.synthsizer = obj_factory(synthsizer_obj)
        self.data_aug = obj_factory(data_aug_obj) if data_aug_obj else None
        self.synth_ratio = synth_ratio
        self.ratio_decay_rate = ratio_decay_rate
        self.min_synth_ratio = min_synth_ratio

    def prepare_synth_input(self, input_data, batch_info):
        synth_on_source = random.random() > 0.5
        source_dict = input_data["source"] if synth_on_source else input_data["target"]
        input_data["target"], synth_info = self.synthsizer(deepcopy(source_dict))
        if self.data_aug is not None:
            input_data["source"], _ =  self.data_aug(source_dict)
        input_data["source"] = source_dict
        batch_info["source_info"] = batch_info["source_info"] if synth_on_source else batch_info["target_info"]
        batch_info["target_info"] = batch_info["source_info"]
        batch_info["pair_name"] = [name + "_and_synth" for name in batch_info["source_info"]["name"]]
        batch_info["synth_info"] = synth_info
        batch_info["is_synth"] = True
        return input_data, batch_info

    def prepare_raw_pair(self, input_data, batch_info):
        if self.data_aug is not None and batch_info["phase"]=="train":
            input_data["source"], _ =  self.data_aug(input_data["source"])
            input_data["target"], _ =  self.data_aug(input_data["target"])
        return input_data, batch_info

    def update_synth_ratio(self,epoch):
        from shapmagn.utils.utils import sigmoid_decay
        cur_synth_ratio = max(sigmoid_decay(epoch, static=20, k=self.ratio_decay_rate)*self.synth_ratio, self.min_synth_ratio)
        return cur_synth_ratio

    def planner(self,phase, current_epoch):
        if phase=="val":
            return True
        elif phase=="debug":
            return False
        elif phase=="train":
            synth_ratio = self.synth_ratio if self.ratio_decay_rate==-1 else self.update_synth_ratio(current_epoch)
            return random.random()< synth_ratio

    def __call__(self,input_data, batch_info):
        use_synth = self.planner(batch_info["phase"],batch_info["epoch"])
        if use_synth:
            return self.prepare_synth_input(input_data,batch_info)
        else:
            return self.prepare_raw_pair(input_data, batch_info)






def prepare_synth_input():
    synthsizer = lung_synth_data()
    def prepare(input_data, batch_info):
        phase = batch_info["phase"]
        if phase in ["train","val"]:
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
        return input_data, batch_info
    return prepare









def create_shape_pair(source, target, toflow=None):
    shape_pair = ShapePair().set_source_and_target(source, target)
    if "toflow" is not None:
        shape_pair.set_to_flow(toflow)
    reg_param = torch.zeros_like(shape_pair.get_control_points(), requires_grad=True)
    shape_pair.set_reg_param(reg_param)
    return shape_pair