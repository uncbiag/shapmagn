from copy import deepcopy
import torch
import torch.nn as nn
from shapmagn.global_variable import Shape
from shapmagn.metrics.losses import Loss
from shapmagn.utils.obj_factory import obj_factory
from shapmagn.utils.utils import sigmoid_decay
from shapmagn.modules.gradient_flow_module import gradient_flow_guide
class DiscreteFlowOPT(nn.Module):
    """
    flow the source via n step, in each step with the #current# source X get updated, the target Y is fixed
    In this class, we provide two approaches to solve this task:

    1. standard spline registration
    for each step a spline registration problem can be defined as
    p^* = argmin_p <p,Kp> + Sim(X+Kp,Y)

    where p is the momentum or registration parameter,
    K is the kernel metric
    an #user-defined# the optimizer is used to solve this problem
    with new  X = X+Kp


    2. gradient flow guided spline registration
    for each step, a rigid kernel regression problem is to be solved:
     p^* = argmin_p <p,Kp> + \lambda \|Kp-v\|^2
    where p is the momentum or registration parameter,  v is the velocity guidence solved from gradient flow defined by wasserstein distance
    this problem can be either solved via standard conjunction gradient solver
    or a fast approximate \tilde{v}=Kv, where K should be Nadaraya-Watson kernel



    """
    def __init__(self, opt):
        super(DiscreteFlowOPT, self).__init__()
        self.opt = opt
        self.gradient_flow_mode =  self.opt[("gradient_flow_mode",False,"if true, work on gradient flow guided spline registration, otherwise standard spline registration")]
        self.drift_every_n_iter = self.opt[("drift_every_n_iter",10, "if n is set bigger than -1, then after every n iteration, the current flowed shape is set as the source shape")]
        spline_kernel_obj = self.opt[("spline_kernel_obj","point_interpolator.nadwat_kernel_interpolator(scale=0.1, exp_order=2)", "shape interpolator in multi-scale solver")]
        self.spline_kernel= obj_factory(spline_kernel_obj)
        interp_kernel_obj = self.opt[("interp_kernel_obj","point_interpolator.nadwat_kernel_interpolator(scale=0.1, exp_order=2)", "kernel for multi-scale interpolation")]
        self.interp_kernel= obj_factory(interp_kernel_obj)
        pair_feature_extractor_obj = self.opt[("pair_feature_extractor_obj", "", "feature extraction function")]
        self.pair_feature_extractor = obj_factory(pair_feature_extractor_obj) if pair_feature_extractor_obj else None
        sim_loss_opt = opt[("sim_loss", {}, "settings for sim_loss_opt")]
        self.sim_loss_fn = Loss(sim_loss_opt)
        self.reg_loss_fn = self.regularization
        self.call_thirdparty_package = False
        self.register_buffer("local_iter", torch.Tensor([0]))
        self.register_buffer("global_iter", torch.Tensor([0]))
        self.print_step = self.opt[('print_step',1,"print every n iteration")]
        self.drift_buffer = {}
        if self.gradient_flow_mode:
            print("in gradient flow mode, points drift every iteration")
            assert "nadwat" in spline_kernel_obj, "in gradient flow mode, spline_kernel should be defined as Nadaraya-Watson"




    def set_loss_fn(self, loss_fn):
        self.sim_loss_fn = loss_fn

    def reset(self):
        self.local_iter = self.local_iter*0



    def flow(self, shape_pair):
        """
        todo  for the rigid kernel regression, anistropic kernel interpolation is not supported yet
        Attention:
        the flow function only get reasonable result if the source and target has similar topology
        If the topology difference between the source to target is large, this flow function is not recommended,
        to workaround, 1. use dense_mode in shape_pair  2. if use multi-scale solver, set the last scale to be -1 with enough iterations
        :param shape_pair:
        :return:
        """
        toflow_points = shape_pair.toflow.points
        control_points = shape_pair.control_points
        control_weights = shape_pair.control_weights

        moving_control_points = self.drift_buffer["moving_control_points"]
        interped_control_points_high = self.interp_kernel(toflow_points, control_points,
                                                          moving_control_points, control_weights)
        interped_control_points_disp = self.spline_kernel(toflow_points, control_points,
                                                               shape_pair.reg_param, control_weights)
        flowed_points = interped_control_points_high + interped_control_points_disp
        flowed = Shape()
        flowed.set_data_with_refer_to(flowed_points,shape_pair.source)
        shape_pair.set_flowed(flowed)
        return shape_pair

    def regularization(self,sm_flow, flow):
        dist = sm_flow * flow
        dist = dist.mean()
        return dist


    def get_factor(self):
        """
        get the regularizer factor according to training strategy

        :return:
        """
        sim_factor = 100
        reg_factor_init =100 #self.initial_reg_factor
        static_epoch = 100
        min_threshold = reg_factor_init/10
        decay_factor = 8
        reg_factor = float(
            max(sigmoid_decay(self.local_iter.item(), static=static_epoch, k=decay_factor) * reg_factor_init, min_threshold))
        return sim_factor, reg_factor


    def update_reg_param_from_low_scale_to_high_scale(self, shape_pair_low, shape_pair_high):
        """todo   do interpolation on disp"""
        #assert self.drift_every_n_iter==-1, "the drift mode is not fully tested in multi-scale mode, disabled for now"
        control_points_high = shape_pair_high.get_control_points()
        control_points_low = shape_pair_low.get_control_points()
        control_weights_low = shape_pair_low.control_weights
        moving_control_points_low = control_points_low if len(self.drift_buffer)==0 else self.drift_buffer["moving_control_points"]
        interped_control_points_high = self.interp_kernel(control_points_high, control_points_low,
                                                     moving_control_points_low, control_weights_low)
        interped_control_points_disp_high = self.spline_kernel(control_points_high,control_points_low,shape_pair_low.reg_param,control_weights_low)
        flowed_control_points_high = interped_control_points_high+interped_control_points_disp_high
        shape_pair_high.set_control_points(control_points_high)
        self.init_reg_param(shape_pair_high)
        self.drift_buffer["moving_control_points"] = flowed_control_points_high.detach().clone()
        return shape_pair_high


    def drift(self, shape_pair):
        """
        Under drift strategy, the source updates during the optimization,  we introduce a "moving" to record the #current# source
        """
        flowed = shape_pair.flowed if self.local_iter>0 else shape_pair.source
        flowed_control_points = self.drift_buffer["moving_control_points"] if self.drift_buffer else  shape_pair.control_points
        flowed_control_points = shape_pair.flowed_control_points if self.local_iter>0 else flowed_control_points
        if self.local_iter % self.drift_every_n_iter==0:
            print("drift at the {} th step, the moving (current source) is updated".format(self.local_iter.item()))
            moving = Shape()
            moving.set_data(points=flowed.points.detach().clone(), weights=flowed.weights.detach().clone())
            self.drift_buffer["moving"] = moving
            self.drift_buffer["moving_control_points"] = flowed_control_points.detach().clone()
            self.reset_reg_param(shape_pair)



    def init_reg_param(self,shape_pair):
        reg_param = torch.zeros_like(shape_pair.get_control_points()).normal_(0, 1e-7)
        reg_param.requires_grad_()
        shape_pair.set_reg_param(reg_param)
        return shape_pair

    def reset_reg_param(self,shape_pair):
        shape_pair.reg_param.data[:] = torch.zeros_like(shape_pair.reg_param).normal_(0, 1e-7)


    def extract_point_fea(self, flowed, target):
        flowed.pointfea = flowed.points
        target.pointfea = target.points
        return flowed, target

    def extract_fea(self, flowed, target):
        """DiscreteFlowOPT supports feature extraction"""
        if not self.pair_feature_extractor:
            return self.extract_point_fea(flowed, target)
        else:
            return self.pair_feature_extractor(flowed,target)



    def standard_spline_forward(self, shape_pair):
        """
        In this forward, the solver should be built either via build_multi_scale_solver / build_single_scale_custom_solver
        reg_param here is the momentum p before kernel operation K,  \tilde{v} = Kp
        :param shape_pair:
        :return:
        """
        if self.drift_every_n_iter == -1:
            control_points = shape_pair.get_control_points()
        else:
            self.drift(shape_pair)
            control_points= self.drift_buffer["moving_control_points"]
        smoothed_reg_param = self.spline_kernel(control_points,control_points,shape_pair.reg_param, shape_pair.control_weights)
        flowed_control_points = control_points + smoothed_reg_param
        shape_pair.set_flowed_control_points(flowed_control_points)
        flowed_has_inferred = shape_pair.infer_flowed()
        shape_pair = self.flow(shape_pair) if not flowed_has_inferred else shape_pair
        shape_pair.flowed, shape_pair.target = self.extract_fea(shape_pair.flowed, shape_pair.target)
        sim_loss = self.sim_loss_fn(shape_pair.flowed, shape_pair.target)
        reg_loss = self.reg_loss_fn(smoothed_reg_param, shape_pair.reg_param)
        sim_factor, reg_factor = self.get_factor()
        sim_loss = sim_loss * sim_factor
        reg_loss = reg_loss * reg_factor
        if self.local_iter % 10 == 0:
            print("{} th step, sim_loss is {}, reg_loss is {}, sim_factor is {}, reg_factor is {}"
                  .format(self.local_iter.item(), sim_loss.item(), reg_loss.item(), sim_factor, reg_factor))
            self.debug(shape_pair)
        loss = sim_loss + reg_loss
        self.local_iter += 1
        self.global_iter +=1
        return loss



    def wasserstein_gradient_flow_guidence(self, flowed, target):
        """
        wassersten gradient flow has a reasonable behavior only when set self.pair_feature_extractor = None
        """
        gradflow_guided_opt = self.opt[("gradflow_guided", {}, "settings for gradflow guidance")]
        gradflow_blur_init = gradflow_guided_opt[
            ("gradflow_blur_init", 0.05, "the inital 'blur' parameter in geomloss setting")]
        update_gradflow_blur_by_raito = gradflow_guided_opt[
            ("update_gradflow_blur_by_raito", 0.05, "the raito that updates the 'blur' parameter in geomloss setting")]
        gradflow_blur_min = gradflow_guided_opt[
            ("gradflow_blur_min", 0.05, "the minium value of the 'blur' parameter in geomloss setting")]
        n_update = self.global_iter.item()
        cur_blur = max(gradflow_blur_init * (update_gradflow_blur_by_raito ** n_update), gradflow_blur_min)
        geomloss_setting = deepcopy(self.opt["sim_loss"]["geomloss"])
        geomloss_setting["geom_obj"] = geomloss_setting["geom_obj"].replace("placeholder", str(cur_blur))
        gradflowed = gradient_flow_guide(flowed, target, geomloss_setting, self.local_iter)
        gradflowed_disp = (gradflowed.points-flowed.points).detach()
        return gradflowed_disp

    def gradflow_spline_foward(self, shape_pair):
        """
       In this forward, the solver should be built via build_single_scale_model_embedded_solver
        gradient flow has a reasonable behavior only when set self.pair_feature_extractor = None
        reg_param here is the momentum p before kernel operation K,  \tilde{v} = Kp
       :param shape_pair:
       :return:
           """
        self.drift(shape_pair)
        moving, control_points = self.drift_buffer["moving"],self.drift_buffer["moving_control_points"]
        moving, shape_pair.target = self.extract_fea(moving, shape_pair.target)
        gradflowed_disp = self.wasserstein_gradient_flow_guidence(moving, shape_pair.target)
        smoothed_reg_param = self.spline_kernel( shape_pair.control_points, shape_pair.control_points, gradflowed_disp, shape_pair.control_weights)
        flowed_control_points = control_points + smoothed_reg_param
        shape_pair.set_flowed_control_points(flowed_control_points)
        flowed_has_inferred = shape_pair.infer_flowed()
        shape_pair = self.flow(shape_pair) if not flowed_has_inferred else shape_pair
        shape_pair.flowed, shape_pair.target = self.extract_fea(shape_pair.flowed, shape_pair.target)
        # the following loss are not used for param update, only used for result analysis
        sim_loss = self.sim_loss_fn(shape_pair.flowed, shape_pair.target)
        reg_loss = self.reg_loss_fn(smoothed_reg_param, shape_pair.reg_param)
        sim_factor, reg_factor = 100, 10
        sim_loss = sim_loss * sim_factor
        reg_loss = reg_loss * reg_factor
        if self.local_iter % 1 == 0:
            print("{} th step, sim_loss is {}, reg_loss is {}, sim_factor is {}, reg_factor is {}"
                  .format(self.local_iter.item(), sim_loss.item(), reg_loss.item(), sim_factor, reg_factor))
            self.debug(shape_pair)
        loss = sim_loss + reg_loss
        self.local_iter += 1
        self.global_iter += 1
        return loss


    def forward(self, shape_pair):
        if self.gradient_flow_mode:
            return self.gradflow_spline_foward(shape_pair)
        else:
            return self.standard_spline_forward(shape_pair)





    def debug(self,shape_pair):
        from shapmagn.utils.visualizer import visualize_point_pair_overlap
        source, flowed, target = shape_pair.source, shape_pair.flowed, shape_pair.target
        # visualize_point_pair_overlap(flowed.points, target.points,
        #                          flowed.weights,target.weights,
        #                          title1="flowed",title2="target", rgb_on=False)
        visualize_point_pair_overlap(flowed.points, target.points,
                                     source.points.squeeze(), target.points.squeeze(),
                                     title1="flowed", title2="target", rgb_on=True)



