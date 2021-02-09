from shapmagn.utils.local_feature_extractor import *

# lung_pair_feature_extractor = pair_feature_extractor


def update_weight(weight_list,iter):
    max_weight = 1
    if iter>0:
        weight_list = [max(0.1*iter+weight,max_weight) for weight in weight_list]
    return weight_list

def lung_pair_feature_extractor(fea_type_list,weight_list=None, radius=0.01, std_normalize=True, include_pos=False):
    fea_extractor = feature_extractor(fea_type_list, radius, std_normalize, include_pos)
    def extract(flowed, target, iter=-1,flowed_gamma=None, target_gamma=None):
        cur_weight_list = update_weight(weight_list, iter) if weight_list is not None else None
        flowed.pointfea, mean, std, _ = fea_extractor(flowed.points,weight_list=cur_weight_list,gamma=flowed_gamma,  return_stats=True)
        target.pointfea, _ = fea_extractor(target.points,weight_list=weight_list,gamma=target_gamma,  mean=mean, std=std)
        return flowed, target
    return extract