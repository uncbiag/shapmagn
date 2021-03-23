from copy import deepcopy
import random
import torch
from shapmagn.shape.point_sampler import  batch_uniform_sampler
from shapmagn.utils.obj_factory import obj_factory
from shapmagn.utils.utils import index_points

class HybirdData(object):
    def __init__(self, synthsizer_obj, data_aug_obj=None, raw_source_target_has_corr = False, corr_sampled_source_target= False, npoints =-1, synth_ratio=1.0, ratio_decay_rate=8, min_synth_ratio=0.3):
        super(HybirdData,self).__init__()
        self.synthsizer = obj_factory(synthsizer_obj) if synthsizer_obj else None
        self.data_aug = obj_factory(data_aug_obj) if data_aug_obj else None
        self.synth_ratio = synth_ratio
        self.ratio_decay_rate = ratio_decay_rate
        self.min_synth_ratio = min_synth_ratio
        self.sampler =None
        self.raw_source_target_has_corr =raw_source_target_has_corr
        self.corr_sampled_source_target =corr_sampled_source_target
        self.npoints = npoints


    def sampling(self, input_data, use_synth):
        sp, sw = input_data["source"]["points"],input_data["source"]["weights"]
        tp, tw = input_data["target"]["points"],input_data["target"]["weights"]

        input_data["source"]["points"], input_data["source"]["weights"], ssind = self.sampler(sp,sw)
        ssind = ssind.long()
        if "pointfea" in input_data["source"]:
            input_data["source"]["pointfea"] = index_points(input_data["source"]["pointfea"], ssind)
        input_data["source"]["extra_info"] = {key:index_points(item,ssind) for key, item in input_data["source"].get("extra_info",{}).items()}
        if use_synth or self.raw_source_target_has_corr:
            gf = tp - sp
            input_data["source"]["extra_info"].update({"gt_flow":index_points(gf,ssind)}) # here we save gt_flow to both source and target, another option is save it to shapepair

        input_data["target"]["points"], input_data["target"]["weights"], tsind = self.sampler(tp, tw)
        tsind = ssind.long() if self.corr_sampled_source_target else tsind.long()
        if "pointfea" in input_data["target"]:
            input_data["target"]["pointfea"] = index_points(input_data["target"]["pointfea"], tsind)
        input_data["extra_info"] = {key:index_points(item,tsind) for key, item in input_data["target"].get("extra_info",{}).items()}
        if "gt_flow" in input_data["source"]["extra_info"]:
            input_data["extra_info"]["gt_flow"] = input_data["source"]["extra_info"]["gt_flow"] # here we save gt_flow to both source and target, another option is save it to shapepair
            input_data["extra_info"]["gt_flowed"] = input_data["extra_info"]["gt_flow"] + input_data["source"]["points"] # here we save gt_flow to both source and target, another option is save it to shapepair
        return input_data

    def non_sampling(self, input_data, use_synth):
        input_data["extra_info"] = input_data.get("extra_info",{})
        if self.raw_source_target_has_corr or use_synth:
            input_data["extra_info"]["gt_flow"] = input_data["target"]["points"] - input_data["source"]["points"]
            input_data["extra_info"]["gt_flowed"] = input_data["target"]["points"]
        return input_data

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
        batch_info["corr_source_target"] = True
        return input_data, batch_info

    def prepare_raw_pair(self, input_data, batch_info):
        if self.data_aug is not None and batch_info["phase"]=="train":
            input_data["source"], _ =  self.data_aug(input_data["source"])
            input_data["target"], _ =  self.data_aug(input_data["target"])
        batch_info["is_synth"] = False
        batch_info["corr_source_target"] = self.raw_source_target_has_corr

        return input_data, batch_info

    def update_synth_ratio(self,epoch):
        from shapmagn.utils.utils import sigmoid_decay
        cur_synth_ratio = max(sigmoid_decay(epoch, static=20, k=self.ratio_decay_rate)*self.synth_ratio, self.min_synth_ratio)
        return cur_synth_ratio

    def planner(self,phase, current_epoch):
        if phase=="val":
            self.sampler = batch_uniform_sampler(self.npoints, fixed_random_seed=True, sampled_by_weight=True)
            return True, True
        elif phase=="debug":
            self.sampler = batch_uniform_sampler(self.npoints, fixed_random_seed=True, sampled_by_weight=True)
            return False, self.corr_sampled_source_target
        elif phase=="train":
            self.sampler = batch_uniform_sampler(self.npoints, fixed_random_seed=False, sampled_by_weight=True)
            synth_ratio = self.synth_ratio if self.ratio_decay_rate==-1 else self.update_synth_ratio(current_epoch)
            return random.random()< synth_ratio, self.corr_sampled_source_target

    def __call__(self,input_data, batch_info):
        use_synth, corr_sampled_source_target = self.planner(batch_info["phase"],batch_info["epoch"])
        if use_synth:
            input_data, batch_info = self.prepare_synth_input(input_data,batch_info)
        else:
            input_data, batch_info = self.prepare_raw_pair(input_data, batch_info)
        if self.npoints>0:
            batch_info["corr_source_target"] = batch_info["corr_source_target"] and corr_sampled_source_target
            return self.sampling(input_data,use_synth), batch_info
        else:
            return self.non_sampling(input_data, use_synth), batch_info
