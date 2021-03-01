from copy import deepcopy
from shapmagn.modules.deep_feature_module import *
from shapmagn.shape.shape_pair_utils import create_shape_pair
from shapmagn.modules.gradient_flow_module import gradient_flow_guide

DEEP_EXTRACTOR = {"pointnet2_extractor": PointNet2FeaExtractor}

class DeepFeature(nn.Module):
    """
    In this class, a deep feature extractor is trained,
    the synth data is used for training
    additionally, a spline model is included for evaluation
    """
    def __init__(self, opt):
        super(DeepFeature, self).__init__()
        self.opt = opt
        create_shape_pair_from_data_dict = opt[
            ("create_shape_pair_from_data_dict", "shape_pair_utils.create_shape_pair_from_data_dict()", "generator func")]
        self.create_shape_pair_from_data_dict = obj_factory(create_shape_pair_from_data_dict)
        decompose_shape_pair_into_dict = opt[
            ("decompose_shape_pair_into_dict", "shape_pair_utils.decompose_shape_pair_into_dict()",
             "decompose shape pair into dict")]
        self.decompose_shape_pair_into_dict = obj_factory(decompose_shape_pair_into_dict)
        spline_kernel_obj = self.opt[("spline_kernel_obj","point_interpolator.NadWatIsoSpline(exp_order=2,kernel_scale=0.05)", "shape interpolator in multi-scale solver")]
        self.spline_kernel= obj_factory(spline_kernel_obj)
        interp_kernel_obj = self.opt[(
        "interp_kernel_obj", "point_interpolator.NadWatIsoSpline(exp_order=2,kernel_scale=0.05)",
        "kernel for multi-scale interpolation")]
        self.interp_kernel = obj_factory(interp_kernel_obj)
        deep_extractor = self.opt[("deep_extractor","pointnet2_extractor","name of deep feature extractor")]
        self.pair_feature_extractor = DEEP_EXTRACTOR[deep_extractor](self.opt[deep_extractor,{},"settings for the deep extractor"])
        self.loss = DeepFeatureLoss(self.opt[("deepfea_loss",{},"settings for deep feature loss")])
        # sim_loss_opt = opt[("sim_loss_for_evaluation_only", {}, "settings for sim_loss_opt, the sim_loss here is not used for training but for evaluation")]
        # self.sim_loss_fn = Loss(sim_loss_opt)
        # self.reg_loss_fn = self.regularization
        self.register_buffer("local_iter", torch.Tensor([0]))
        self.print_step = self.opt[('print_step',1,"print every n iteration")]

    def check_if_update_lr(self):
         return  None, None




    def set_cur_epoch(self, cur_epoch):
        pass
 

    def set_loss_fn(self, loss_fn):
        self.sim_loss_fn = loss_fn

    def reset(self):
        self.local_iter = self.local_iter*0



    def model_eval(self, input_data, batch_info=None):
        """
        for  deep approach, we assume the source points = control points
        :param shape_pair:
        :param batch_info:
        :return:
        """
        if batch_info["is_synth"]:
            loss, shape_data_dict = self.forward(input_data)
            shape_pair = self.create_shape_pair_from_data_dict(shape_data_dict)

        else:
            shape_pair = self.create_shape_pair_from_data_dict(input_data)
            shape_pair.source, shape_pair.target = self.pair_feature_extractor(shape_pair.source, shape_pair.target)
            loss = -1

        geomloss_setting = deepcopy(self.opt["deepfea_loss"]["geomloss"])
        geomloss_setting.print_settings_off()
        geomloss_setting["mode"] = "analysis"
        geomloss_setting["attr"] = "pointfea"
        mapped_target_index,mapped_topK_target_index, mapped_position = wasserstein_forward_mapping(shape_pair.source, shape_pair.target,
                                                                           geomloss_setting)  # BxN
        source_points = shape_pair.source.points
        source_weights = shape_pair.source.weights
        disp = mapped_position - source_points
        smoothed_disp = self.spline_kernel(source_points, source_points,disp,
                                                source_weights)
        flowed_points = source_points + smoothed_disp
        shape_pair.flowed = Shape().set_data_with_refer_to(flowed_points, shape_pair.source)
        B, N = source_points.shape[0], source_points.shape[1]
        device = source_points.device
        if batch_info["is_synth"]:
            # compute mapped acc
            gt_index = torch.arange(N, device=device).repeat(B,1)
            acc = (mapped_target_index==gt_index).sum(1)/N
        else:
            acc =torch.tensor([-1],device=device).repeat(B)
        metrics= {"score": [_acc.item() for _acc in acc], "loss": [_loss.item() for _loss in loss]}
        return metrics, self.decompose_shape_pair_into_dict(shape_pair)




    def forward(self, input_data):
        shape_pair = self.create_shape_pair_from_data_dict(input_data)
        moving, shape_pair.target = self.pair_feature_extractor(shape_pair.source, shape_pair.target)
        loss = self.loss(moving, shape_pair.target)
        shape_pair.source = moving
        return loss, self.decompose_shape_pair_into_dict(shape_pair)


