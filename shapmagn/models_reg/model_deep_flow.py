from shapmagn.modules_reg.module_gradflow_prealign import GradFlowPreAlign
from shapmagn.modules_reg.module_gradient_flow import point_based_gradient_flow_guide
from shapmagn.modules_reg.module_teaser import Teaser
from shapmagn.utils.utils import sigmoid_decay
from shapmagn.modules_reg.module_deep_flow import *
#from pytorch_memlab import profile

DEEP_REGPARAM_GENERATOR= {
    "flownet_regparam": DeepFlowNetRegParam,
    "pwcnet_regparam":PointConvSceneFlowPWCRegParam,
    "flotnet_regparam":FLOTRegParam,
}
Deep_Loss = {"deepflow_loss":DeepFlowLoss, "pwc_loss":PWCLoss}


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
        loss_name = self.opt[("deep_loss", "deepflow_loss", "name of deep loss")]
        self.use_prealign = self.opt[("use_prealign", False, "prealign the shape first")]
        if self.use_prealign:
            self.init_prealign()
        self.deep_regparam_generator = DEEP_REGPARAM_GENERATOR[generator_name](
            self.opt[generator_name, {}, "settings for the deep registration parameter generator"])
        self.flow_model = FlowModel(self.opt["flow_model",{},"settings for the flow model"])
        self.loss = Deep_Loss[loss_name](self.opt[loss_name, {}, "settings for the deep registration parameter generator"])
        self.geom_loss_opt_for_eval = opt[("geom_loss_opt_for_eval", {}, "settings for sim_loss_opt, the sim_loss here is not used for training but for evaluation")]
        aniso_post_kernel_obj = opt[("aniso_post_kernel_obj","", "shape interpolator")]
        self.aniso_post_kernel = obj_factory(aniso_post_kernel_obj) if aniso_post_kernel_obj else None
        self.register_buffer("local_iter", torch.Tensor([0]))
        self.n_step = opt[("n_step",1,"number of iteration step")]
        self.step_weight_list = opt[("step_weight_list",[1/self.n_step]*self.n_step,"weight for each step")]
        external_evaluate_metric_obj = self.opt[("external_evaluate_metric_obj", "","external evaluate metric")]
        self.external_evaluate_metric = obj_factory(external_evaluate_metric_obj) if external_evaluate_metric_obj else None
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
        self.buffer={}

    def flow(self, shape_pair):
        """
        if the LDDMM is used, we assume the nstep=1
        :param shape_pair:
        :return:
        """
        if self.use_prealign:
            toflow_points = shape_pair.toflow.points
            prealigned_toflow_points = self.apply_prealign_transform(self.buffer["prealign_param"],toflow_points)
            prealigned_toflow = Shape().set_data_with_refer_to(prealigned_toflow_points,shape_pair.toflow)
            shape_pair.toflow = prealigned_toflow
        if self.flow_model.model_type=="lddmm":
            for s in range(self.n_step):
                shape_pair.reg_param = self.buffer["reg_param_step{}".format(s)]
                flowed_control_points,flowed_points = self.flow_model.flow(shape_pair)
                shape_pair.control_points = flowed_control_points
                shape_pair.toflow.points = flowed_points
        else:
            flowed_points = self.flow_model.flow(shape_pair)
        flowed = Shape()
        flowed.set_data_with_refer_to(flowed_points, shape_pair.toflow)
        shape_pair.set_flowed(flowed)
        return shape_pair

    def model_eval(self, input_data, batch_info=None):
        """
        for  deep approach, we assume the source points = control points
        :param shape_pair:
        :param batch_info:
        :return:

        """
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        loss, shape_data_dict = self.forward(input_data, batch_info)
        shape_pair = self.create_shape_pair_from_data_dict(shape_data_dict)
        corr_source_target = batch_info["corr_source_target"]

        # end.record()
        # torch.cuda.synchronize()
        # running_time = start.elapsed_time(end)
        # print("{}, it takes {} ms".format(shape_pair.pair_name, running_time))

        geomloss_setting = deepcopy(self.geom_loss_opt_for_eval)
        geomloss_setting.print_settings_off()
        geomloss_setting["mode"] = "analysis"
        geomloss_setting["attr"] = "points"
        use_bary_map = geomloss_setting[("use_bary_map",True,"take wasserstein baryceneter if there is little noise or outlier,  otherwise use gradient flow")]
        #in the first epoch, we would output the ot baseline, this is for analysis propose, comment the following two lines if don't needed
        source_points = shape_pair.source.points
        # if self.cur_epoch==0:
        #     print("In the first epoch, the validation/debugging output is the baseline ot mapping")
        #     shape_pair.flowed= Shape().set_data_with_refer_to(source_points,shape_pair.source)
        if use_bary_map:
            mapped_target_index,mapped_topK_target_index, mapped_position = wasserstein_barycenter_mapping(shape_pair.flowed, shape_pair.target, geomloss_setting)  # BxN
        else:
            mapped_position, wasserstein_dist = point_based_gradient_flow_guide(shape_pair.flowed, shape_pair.target, geomloss_setting)
        if self.aniso_post_kernel is not None:
            disp = mapped_position - shape_pair.flowed.points
            flowed_points = shape_pair.flowed.points
            smoothed_disp = self.aniso_post_kernel(flowed_points, flowed_points, disp, shape_pair.flowed.weights)
            mapped_position = flowed_points +smoothed_disp
            # todo experimental code,  clean and wrap the code into a iter function
            #
            cur_flowed = Shape().set_data_with_refer_to(mapped_position,shape_pair.flowed)
            mapped_target_index, mapped_topK_target_index, mapped_position = wasserstein_barycenter_mapping(
                cur_flowed, shape_pair.target, geomloss_setting)  # BxN
            disp = mapped_position - cur_flowed.points
            flowed_points = cur_flowed.points
            smoothed_disp = self.aniso_post_kernel(flowed_points, flowed_points, disp, shape_pair.flowed.weights)
            mapped_position = flowed_points + smoothed_disp
            # cur_flowed = Shape().set_data_with_refer_to(mapped_position, shape_pair.flowed)
            # mapped_target_index, mapped_topK_target_index, mapped_position = wasserstein_barycenter_mapping(
            #     cur_flowed, shape_pair.target, geomloss_setting)  # BxN
            # disp = mapped_position - cur_flowed.points
            # flowed_points = cur_flowed.points
            # smoothed_disp = self.aniso_post_kernel(flowed_points, flowed_points, disp, shape_pair.flowed.weights)
            # mapped_position = flowed_points + smoothed_disp

        end.record()
        torch.cuda.synchronize()
        running_time =  start.elapsed_time(end)
        print("{}, it takes {} ms".format(shape_pair.pair_name, running_time))
        if use_bary_map:
            #   since barycenter mapping doesn't return wasserstein distance ,here we compute the wasserstein distance
            #  todo avoid dupilcate computation, directly compute wasserstein distance in from dual embedding in barycenter function
            _, wasserstein_dist = point_based_gradient_flow_guide(shape_pair.flowed, shape_pair.target, geomloss_setting)

        B, N = source_points.shape[0], source_points.shape[1]
        device = source_points.device
        print("the current data is {}".format("synth" if batch_info["is_synth"] else "real"))
        if corr_source_target:
            gt_index = torch.arange(N, device=device).repeat(B, 1)  #B,N
            acc = (mapped_target_index == gt_index).sum(1) / N
            topk_acc = ((mapped_topK_target_index == (gt_index[...,None])).sum(2) >0).sum(1)/N
            metrics = {"score": [_topk_acc.item() for _topk_acc in topk_acc], "loss": [_loss.item() for _loss in loss],
                       "_acc":[_acc.item() for _acc in acc], "topk_acc":[_topk_acc.item() for _topk_acc in topk_acc],
                       "ot_dist":[_ot_dist.item() for _ot_dist in wasserstein_dist],"forward_t":[running_time/B]*B}
        else:
            metrics = {"score": [_sim.item() for _sim in self.buffer["sim_loss"]], "loss": [_loss.item() for _loss in loss],
                       "ot_dist":[_ot_dist.item() for _ot_dist in wasserstein_dist],"forward_t":[running_time/B]*B}
        if self.external_evaluate_metric is not None:
            additional_param = {"model":self, "initial_nonp_control_points":self.buffer["initial_nonp_control_points"],"prealign_param":self.buffer["prealign_param"],"prealigned":self.buffer["prealigned"]}
            self.external_evaluate_metric(metrics, shape_pair,batch_info,additional_param =additional_param, alias="")
            additional_param.update({"mapped_position": mapped_position})
            self.external_evaluate_metric(metrics, shape_pair,batch_info, additional_param, "_gf")
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

    def init_prealign(self):
        prealign_opt = self.opt[("prealign_opt",{},"settings for prealign")]
        prealign_module_dict = {"teaser": Teaser, "gradflow_prealign": GradFlowPreAlign}  # "probreg":ProbReg,
        self.prealign_module_type = prealign_opt[("module_type", "probreg", "lddmm module type: teaser")]
        self.prealign_module = prealign_module_dict[self.prealign_module_type](
            prealign_opt[(self.prealign_module_type, {}, "settings for prealign module")])
        self.prealign_module.set_mode("prealign")
    def apply_prealign_transform(self,prealign_param, points):
        """
        :param prealign_param: Bx(D+1)xD: BxDxD transfrom matrix and Bx1xD translation
        :param points: BxNxD
        :return:
        """
        dim = points.shape[-1]
        points = torch.bmm(points, prealign_param[:, :dim, :])
        points = prealign_param[:, dim:, :].contiguous() + points
        return points

    def prealign(self,shape_pair):

        with torch.no_grad():
            source = shape_pair.source
            target = shape_pair.target
            prealign_param = self.prealign_module(source, target, shape_pair.reg_param)
            self.buffer={"prealign_param" : prealign_param.clone().detach()}
            flowed_points = self.apply_prealign_transform(prealign_param, source.points)
            self.buffer.update({"prealigned" : Shape().set_data_with_refer_to(flowed_points, source,detach=True)})
            moving = Shape().set_data_with_refer_to(flowed_points, source)
        return moving

    #@profile
    def forward(self, input_data, batch_info=None):
        """
       :param shape_pair:
       :return:
           """
        sim_factor, reg_factor,reg_param_scale = self.get_factor()
        shape_pair = self.create_shape_pair_from_data_dict(input_data)
        self.buffer = {"prealign_param": None,"prealigned": None}
        moving = shape_pair.source if not self.use_prealign else self.prealign(shape_pair)
        has_gt = batch_info["has_gt"]
        gt_flowed = Shape()
        if has_gt:
            gt_flowed_points = shape_pair.extra_info["gt_flowed"]
            gt_flowed.set_data_with_refer_to(gt_flowed_points,moving)
        sim_loss, reg_loss = 0, 0
        debug_reg_param_list = []
        for s in range(self.n_step):
            shape_pair, additional_param = self.deep_regparam_generator(moving, shape_pair) # todo control points is initialized in deep module, but should be exported externally in furture
            debug_reg_param_list.append(shape_pair.reg_param.abs().mean())
            shape_pair.reg_param = shape_pair.reg_param*reg_param_scale
            flowed, _reg_loss = self.flow_model(moving, shape_pair, additional_param)
            if s==0:
                self.buffer ["initial_nonp_control_points"]=shape_pair.control_points.clone().detach()
            self.buffer["reg_param_step{}".format(s)]= shape_pair.reg_param.clone().detach()
            additional_param.update({"source":shape_pair.source, "moving":moving})
            sim_loss += self.step_weight_list[s]*self.loss(flowed,shape_pair.target,gt_flowed,has_gt = has_gt,additional_param=additional_param)
            reg_loss += self.step_weight_list[s]*_reg_loss
            moving = Shape().set_data_with_refer_to(flowed.points.clone().detach(), flowed)
        shape_pair.flowed = flowed
        self.buffer["sim_loss"] = sim_loss.detach()
        self.buffer["reg_loss"] = reg_loss.detach()
        sim_loss = sim_loss * sim_factor
        reg_loss = reg_loss * reg_factor
        if self.local_iter % self.print_step == 0:
            # if debug_reg_param<-5 or debug_reg_param>5:
            #     print("the  average abs mean of the  average abs mean of the reg_param is {}, please adjust the 'reg_param_scale', best make it in [-1,1]".format(debug_reg_param))
            # else:
            #     print("the average abs mean of the reg_param is {}, best in [-1,1]".format(debug_reg_param))
            print("the average abs mean of the reg_param is {}, best in range [-1,1]".format(debug_reg_param_list))
            print("{} th step, {} sim_loss is {}, reg_loss is {}, sim_factor is {}, reg_factor is {}"
                  .format(self.local_iter.item(),"synth_data" if batch_info["is_synth"] else "real_data", sim_loss.mean().item(), reg_loss.mean().item(), sim_factor, reg_factor))
        loss = sim_loss + reg_loss
        self.local_iter += 1
        return loss, self.decompose_shape_pair_into_dict(shape_pair)





