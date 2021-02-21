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
    aniso_kernel_scale = 0.03
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


class LungFeatureExtractor(object):
    def __init__(self,fea_type_list,weight_list=None, radius=0.01,get_anistropic_gamma_obj=None, std_normalize=True, include_pos=False,fixed=False):
        """

        :param fea_type_list:
        :param weight_list:
        :param radius:  used only for isotropic kernel
        :param get_anistropic_gamma_obj: a function object to compute anisotropic gamma
        :param std_normalize:
        :param include_pos:
        :param fixed:
        """
        self.fea_type_list = fea_type_list
        self.feature_extractor = feature_extractor(fea_type_list, radius, std_normalize, include_pos=False)
        self.weight_list = weight_list
        self.radius = radius
        self.get_anistropic_gamma = None if get_anistropic_gamma_obj is None else partial_obj_factory(get_anistropic_gamma_obj)
        self.std_normalize = std_normalize
        self.include_pos = include_pos
        self.fixed = fixed
        self.buffer = {"flowed_gamma":None, "target_gamma":None}
        self.iter = 0

    def update_weight(self, weight_list, iter):
        max_weight = 1
        if iter > 2:
            weight_list = [min(max(0.1 * iter + weight, max_weight), 0.5) for weight in weight_list]
        return weight_list

    def __call__(self,flowed, target, iter=-1):
        flowed_gamma = None
        target_gamma = None
        if self.buffer["flowed_gamma"] is None and self.get_anistropic_gamma is not None:
            flowed_gamma = self.get_anistropic_gamma(flowed.points)
        if self.buffer["target_gamma"] is None and self.get_anistropic_gamma is not None:
            target_gamma = self.get_anistropic_gamma(target.points)
        cur_weight_list = update_weight(self.weight_list, iter) if self.weight_list is not None else None
        if not self.fixed or self.iter==0:
            flowed_pointfea, mean, std, _ = self.feature_extractor(flowed.points,flowed.weights, weight_list=cur_weight_list, gamma=flowed_gamma, return_stats=True)
            target_pointfea, _ = self.feature_extractor(target.points,target.weights, weight_list=cur_weight_list, gamma=target_gamma, mean=mean, std=std)
            self.buffer["flowed_pointfea"] = flowed_pointfea
            self.buffer["target_pointfea"] = target_pointfea
        elif self.fixed and self.iter>0:
            flowed_pointfea = self.buffer["flowed_pointfea"]
            target_pointfea = self.buffer["target_pointfea"]
        if self.include_pos:
            flowed_pointfea = torch.cat([flowed.points,flowed_pointfea],-1)
            target_pointfea = torch.cat([target.points,target_pointfea],-1)
        if self.include_weight:
            flowed_pointfea = torch.cat([flowed.weights,flowed_pointfea],-1)
            target_pointfea = torch.cat([target.weights,target_pointfea],-1)

        flowed.pointfea = flowed_pointfea
        target.pointfea = target_pointfea
        self.iter += 1
        return flowed, target

