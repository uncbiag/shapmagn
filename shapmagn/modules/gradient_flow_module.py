from shapmagn.global_variable import Shape
from shapmagn.metrics.losses import GeomDistance
from torch.autograd import grad
def gradient_flow_guide(flowed,target,geomloss_setting, local_iter, feature_extractor=None):
    # todo check if the feature extractor.e.g. affine like transform, should be disabled
    geomloss = GeomDistance(geomloss_setting)
    flowed_points_clone = flowed.points.detach().clone()
    flowed_points_clone.requires_grad_()
    flowed_clone = Shape()
    flowed_clone.set_data_with_refer_to(flowed_points_clone,
                                        flowed)  # shallow copy, only points are cloned, other attr are not
    loss = geomloss(flowed_clone, target)
    print("{} th step, before gradient flow, the ot distance between the flowed and the target is {}".format(
        local_iter.item(), loss.item()))
    grad_flowed_points = grad(loss, flowed_points_clone)[0]
    flowed_points_clone = flowed_points_clone - grad_flowed_points / flowed_clone.weights
    flowed_clone.points = flowed_points_clone.detach()
    loss = geomloss(flowed_clone, target)
    print(
        "{} th step, after gradient flow, the ot distance between the gradflowed guided points and the target is {}".format(
            local_iter.item(), loss.item()))
    return flowed_clone