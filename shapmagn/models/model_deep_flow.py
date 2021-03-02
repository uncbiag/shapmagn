from shapmagn.utils.utils import sigmoid_decay
from shapmagn.modules.deep_flow_module import *
from shapmagn.shape.shape_pair_utils import create_shape_pair

DEEP_REGPARAM_GENERATOR= {"flownet_regparam": DeepRegParm}


class DeepDiscreteFlow(nn.Module):
    """
    flow the source via n step, in each step with the #current# source X get updated, the target Y is fixed

    """
    def __init__(self, opt):
        super(DeepDiscreteFlow, self).__init__()
        self.opt = opt
        create_shape_pair_from_data_dict = opt[
            ("create_shape_pair_from_data_dict", "shape_pair_utils.create_shape_pair_from_data_dict()", "generator func")]
        self.create_shape_pair_from_data_dict = obj_factory(create_shape_pair_from_data_dict)
        decompose_shape_pair_into_dict = opt[
            ("decompose_shape_pair_into_dict", "shape_pair_utils.decompose_shape_pair_into_dict()",
             "decompose shape pair into dict")]
        self.decompose_shape_pair_into_dict = obj_factory(decompose_shape_pair_into_dict)
        generator_name = self.opt[("deep_regparam_generator", "flownet_regparam", "name of deep deep_regparam_generator")]
        self.deep_regparam_generator = DEEP_REGPARAM_GENERATOR[generator_name](
            self.opt[generator_name, {}, "settings for the deep registration parameter generator"])
        self.flow_model = FlowModel(self.opt["flow_model",{},"settings for the flow model"])
        self.loss = DeepFlowLoss(self.opt[("deepflow_loss", {}, "settings for deep flow loss")])
        # sim_loss_opt = opt[("sim_loss_for_evaluation_only", {}, "settings for sim_loss_opt, the sim_loss here is not used for training but for evaluation")]
        # self.sim_loss_fn = Loss(sim_loss_opt)
        # self.reg_loss_fn = self.regularization
        self.register_buffer("local_iter", torch.Tensor([0]))
        self.print_step = self.opt[('print_step',1,"print every n iteration")]
        self.buffer = {}

    def check_if_update_lr(self):
        return None, None

    def set_cur_epoch(self, cur_epoch):
        self.cur_epoch = cur_epoch


    def set_loss_fn(self, loss_fn):
        self.sim_loss_fn = loss_fn

    def reset(self):
        self.local_iter = self.local_iter*0



    def flow(self, input_data):
        """
        The deep method doesn't support control points, and assume source.points = control_points
        but we remain 'flow' method for ambient interpolation

        :param shape_pair:
        :return:
        """
        shape_pair = self.create_shape_pair_from_data_dict(input_data)
        toflow_points = shape_pair.toflow.points
        control_points = shape_pair.control_points
        control_weights = shape_pair.control_weights

        moving_control_points = self.drift_buffer["moving_control_points"]
        interped_control_points = self.self.flow_model.interp(toflow_points, control_points,
                                                          moving_control_points, control_weights)
        flowed_points =interped_control_points
        flowed = Shape()
        flowed.set_data_with_refer_to(flowed_points,shape_pair.source)
        shape_pair.set_flowed(flowed)
        return self.decompose_shape_pair_into_dict(shape_pair)

    def model_eval(self, input_data, batch_info=None):
        """
        for  deep approach, we assume the source points = control points
        :param shape_pair:
        :param batch_info:
        :return:
        """
        loss, shape_data_dict = self.forward(input_data)
        shape_pair = self.create_shape_pair_from_data_dict(shape_data_dict)
        geomloss_setting = deepcopy(self.opt["deepflow_loss"]["geomloss"])
        geomloss_setting.print_settings_off()
        geomloss_setting["mode"] = "analysis"
        geomloss_setting["attr"] = "points"
        mapped_target_index,mapped_topK_target_index, mapped_position = wasserstein_forward_mapping(shape_pair.flowed, shape_pair.target,
                                                                           geomloss_setting)  # BxN
        wasserstein_dist = self.loss.geom_loss(shape_pair.flowed, shape_pair.target)
        source_points = shape_pair.source.points
        B, N = source_points.shape[0], source_points.shape[1]
        device = source_points.device
        print("debugging, synth is {}".format( batch_info["is_synth"]))
        if batch_info["is_synth"]:
            # compute mapped acc
            gt_index = torch.arange(N, device=device).repeat(B, 1)  #B,N
            acc = (mapped_target_index == gt_index).sum(1) / N
            topk_acc = ((mapped_topK_target_index == (gt_index[...,None])).sum(2) >0).sum(1)/N
            metrics = {"score": [_acc.item() for _acc in acc], "loss": [_loss.item() for _loss in loss],
                       "_acc":[_acc.item() for _acc in acc], "topk_acc":[_topk_acc.item() for _topk_acc in topk_acc],
                       "ot_dist":[_ot_dist.item() for _ot_dist in wasserstein_dist]}
        else:
            metrics = {"score": [_sim.item() for _sim in self.buffer["sim_loss"]], "loss": [_loss.item() for _loss in loss],
                       "ot_dist":[_ot_dist.item() for _ot_dist in wasserstein_dist]}
        return metrics, self.decompose_shape_pair_into_dict(shape_pair)



    def get_factor(self):
        """
        get the regularizer factor according to training strategy

        :return:
        """
        sim_factor = self.opt[("sim_factor",1,"similarity factor")]
        reg_factor_init=self.opt[("reg_factor_init",10,"initial regularization factor")]
        reg_factor_decay = self.opt[("reg_factor_decay",5,"regularization decay factor")]
        reg_param_scale = self.opt[("reg_param_scale",1,"reg param factor to adjust the reg param scale")]
        static_epoch = self.opt[("static_epoch",1,"first # epoch the factor doesn't change")]
        min_threshold = reg_factor_init/10
        reg_factor = float(
            max(sigmoid_decay(self.cur_epoch, static=static_epoch, k=reg_factor_decay) * reg_factor_init, min_threshold))
        return sim_factor, reg_factor, reg_param_scale

    def forward(self, input_data):
        """
       :param shape_pair:
       :return:
           """
        shape_pair = self.create_shape_pair_from_data_dict(input_data)
        reg_param = self.deep_regparam_generator(shape_pair.source, shape_pair.target)
        sim_factor, reg_factor,reg_param_scale = self.get_factor()
        debug_reg_param = reg_param.abs().mean()
        reg_param = reg_param*reg_param_scale
        shape_pair.source.points = shape_pair.source.points.detach()
        flowed, reg_loss = self.flow_model(shape_pair.source, reg_param)
        shape_pair.flowed = flowed
        sim_loss = self.loss(flowed, shape_pair.target)
        self.buffer["sim_loss"] = sim_loss.detach()
        self.buffer["reg_loss"] = reg_loss.detach()
        sim_loss = sim_loss * sim_factor
        reg_loss = reg_loss * reg_factor
        if self.local_iter % self.print_step == 0:
            if debug_reg_param<-5 or debug_reg_param>5:
                print("the  average abs mean of the  average abs mean of the reg_param is {}, please adjust the 'reg_param_scale', best make it in [-1,1]".format(debug_reg_param))
            else:
                print("the average abs mean of the reg_param is {}, best in [-1,1]".format(debug_reg_param))

            print("{} th step, sim_loss is {}, reg_loss is {}, sim_factor is {}, reg_factor is {}"
                  .format(self.local_iter.item(), sim_loss.mean().item(), reg_loss.mean().item(), sim_factor, reg_factor))
        loss = sim_loss + reg_loss
        self.local_iter += 1
        return loss, self.decompose_shape_pair_into_dict(shape_pair)





