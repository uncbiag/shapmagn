from copy import deepcopy
import torch

from shapmagn.metrics.losses import GeomDistance
from shapmagn.modules.gradient_flow_module import wasserstein_forward_mapping


def opt_flow_model_eval(shape_pair, batch_info=None,geom_loss_opt_for_eval=None,external_evaluate_metric=None):
    """
    for  deep approach, we assume the source points = control points
    :param shape_pair:
    :param batch_info:
    :param geom_loss_opt_for_eval:
    :return:
    """
    corr_source_target = batch_info["corr_source_target"]
    geomloss_setting = deepcopy(geom_loss_opt_for_eval)
    geomloss_setting.print_settings_off()
    geomloss_setting["mode"] = "analysis"
    geomloss_setting["attr"] = "points"

    mapped_target_index, mapped_topK_target_index, mapped_position = wasserstein_forward_mapping(shape_pair.flowed,
                                                                                                 shape_pair.target,
                                                                                                 geomloss_setting)  # BxN
    geom_loss = GeomDistance(geomloss_setting)
    wasserstein_dist = geom_loss(shape_pair.flowed, shape_pair.target)
    source_points = shape_pair.source.points
    B, N = source_points.shape[0], source_points.shape[1]
    device = source_points.device
    if corr_source_target:
        # compute mapped acc
        gt_index = torch.arange(N, device=device).repeat(B, 1)  # B,N
        acc = (mapped_target_index == gt_index).sum(1) / N
        topk_acc = ((mapped_topK_target_index == (gt_index[..., None])).sum(2) > 0).sum(1) / N
        metrics = {"score": [_topk_acc.item() for _topk_acc in topk_acc],
                   "_acc": [_acc.item() for _acc in acc], "topk_acc": [_topk_acc.item() for _topk_acc in topk_acc],
                   "ot_dist": [_ot_dist.item() for _ot_dist in wasserstein_dist]}
    else:
        metrics = {"score": [_ot_dist.item() for _ot_dist in wasserstein_dist],
                   "ot_dist": [_ot_dist.item() for _ot_dist in wasserstein_dist]}
    if external_evaluate_metric is not None:
        external_evaluate_metric(metrics, shape_pair, batch_info, additional_param=None, alias="")
        external_evaluate_metric(metrics, shape_pair, batch_info, {"mapped_position": mapped_position}, "_and_gf")
    return metrics, shape_pair