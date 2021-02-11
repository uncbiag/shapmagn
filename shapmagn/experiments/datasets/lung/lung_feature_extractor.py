from shapmagn.utils.local_feature_extractor import *
from shapmagn.utils.obj_factory import *

# lung_pair_feature_extractor = pair_feature_extractor


def update_weight(weight_list,iter):
    max_weight = 1
    if iter>2:
        weight_list = [min(max(0.1*iter+weight,max_weight),0.5) for weight in weight_list]
    return weight_list

def lung_pair_feature_extractor(fea_type_list,weight_list=None, radius=0.01, std_normalize=True, include_pos=False):
    fea_extractor = feature_extractor(fea_type_list, radius, std_normalize, include_pos)
    aniso_kernel_scale = 0.08
    get_anistropic_gamma_obj ="local_feature_extractor.compute_anisotropic_gamma_from_points(cov_sigma_scale=0.02,aniso_kernel_scale={},principle_weight=(2.,1.,1.),eigenvalue_min=0.1,iter_twice=True)".format(aniso_kernel_scale)
    get_anistropic_gamma = partial_obj_factory(get_anistropic_gamma_obj)
    def extract(flowed, target, iter=-1,flowed_gamma=None, target_gamma=None):
        if flowed_gamma is None:
            flowed_gamma =get_anistropic_gamma(flowed.points)
        if target_gamma is None:
            target_gamma =get_anistropic_gamma(target.points)
        cur_weight_list = update_weight(weight_list, iter) if weight_list is not None else None
        flowed.pointfea, mean, std, _ = fea_extractor(flowed.points,weight_list=cur_weight_list,gamma=flowed_gamma,  return_stats=True)
        target.pointfea, _ = fea_extractor(target.points,weight_list=weight_list,gamma=target_gamma,  mean=mean, std=std)
        return flowed, target
    return extract