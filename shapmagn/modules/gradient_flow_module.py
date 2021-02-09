from copy import deepcopy
import torch
from shapmagn.global_variable import Shape
from shapmagn.utils.obj_factory import obj_factory
from shapmagn.metrics.losses import GeomDistance
from torch.autograd import grad

def positional_based_gradient_flow_guide(cur_source,target,geomloss_setting, local_iter):
    geomloss_setting = deepcopy(geomloss_setting)
    geomloss_setting['attr'] = "points"
    geomloss = GeomDistance(geomloss_setting)
    cur_source_points_clone = cur_source.points.detach().clone()
    cur_source_points_clone.requires_grad_()
    cur_source_clone = Shape()
    cur_source_clone.set_data_with_refer_to(cur_source_points_clone,
                                        cur_source)  # shallow copy, only points are cloned, other attr are not
    loss = geomloss(cur_source_clone, target)
    print("{} th step, before gradient flow, the ot distance between the cur_source and the target is {}".format(
        local_iter.item(), loss.item()))
    grad_cur_source_points = grad(loss, cur_source_points_clone)[0]
    cur_source_points_clone = cur_source_points_clone - grad_cur_source_points / cur_source_clone.weights
    cur_source_clone.points = cur_source_points_clone.detach()
    loss = geomloss(cur_source_clone, target)
    print(
        "{} th step, after gradient flow, the ot distance between the gradflowed guided points and the target is {}".format(
            local_iter.item(), loss.item()))
    return cur_source_clone






def wasserstein_forward_mapping(cur_source, target,gemloss_setting,local_iter=None):
    from pykeops.torch import LazyTensor
    geom_obj = gemloss_setting["geom_obj"].replace(")", ",potentials=True)")
    blur_arg_filtered = filter(lambda x: "blur" in x, geom_obj.split(","))
    blur = eval(list(blur_arg_filtered)[0].replace("blur", "").replace("=", ""))
    p = gemloss_setting[("p", 2,"cost order")]
    mode = gemloss_setting[("mode", 'hard',"soft, hard")]
    confid = gemloss_setting[("confid", 0.1,"cost order")]
    geomloss = obj_factory(geom_obj)
    attr = "pointfea"
    attr1 = getattr(cur_source, attr).detach()
    attr2 = getattr(target, attr).detach()
    points1 = cur_source.points
    points2 = target.points
    weight1 = cur_source.weights[:, :, 0]  # remove the last dim
    weight2 = target.weights[:, :, 0]  # remove the last dim
    F_i, G_j = geomloss(weight1, attr1, weight2,
                        attr2)  # todo batch sz of input and output in geomloss is not consistent

    B, N, M, D = points1.shape[0], points1.shape[1], points2.shape[1], points2.shape[2]
    a_i, x_i = LazyTensor(cur_source.weights.view(B,N, 1, 1)), LazyTensor(attr1.view(B,N, 1, -1))
    b_j, y_j = LazyTensor(target.weights.view(B,1, M, 1)), LazyTensor(attr2.view(B,1, M, -1))
    F_i, G_j = LazyTensor(F_i.view(B,N, 1, 1)), LazyTensor(G_j.view(B,1, M, 1))
    C_ij = (1 / p) * ((x_i - y_j) ** p).sum(-1)  # (B,N,M,1) cost matrix
    eps = blur ** p  # temperature epsilon
    P_i = ((F_i + G_j - C_ij) / eps).exp() * (b_j)  # (B, N,M,1) transport plan
    if mode=="soft":
        position_to_map = LazyTensor(points2.view(B,1, M, -1))  # B,1xMxD
        mapped_position = (P_i*position_to_map).sum_reduction(2) #(B,N,M,D)-> (B,N,D)
    elif mode == "hard":
        P_i_max, P_i_index = P_i.max_argmax(2) #  over M,  return (B,N)
        pos_batch_list = []
        for b in range(B):
            pos_batch_list.append(points2[b,P_i_index[b,:,0]])
        mapped_position = torch.stack(pos_batch_list,0)
    else:
        raise ValueError("mode {} not defined, support: soft/ hard/ confid".format(mode))
    print("OT based forward mapping complete")
    mapped_shape = Shape()
    mapped_shape.set_data_with_refer_to(mapped_position,cur_source)
    return mapped_shape


def gradient_flow_guide(mode="grad_forward"):
    postion_based = mode =="grad_forward"
    def guide(cur_source,target,geomloss_setting, local_iter=None):
        if postion_based:
            return positional_based_gradient_flow_guide(cur_source, target, geomloss_setting, local_iter)
        else:
            return wasserstein_forward_mapping(cur_source, target, geomloss_setting, local_iter)
    return guide






