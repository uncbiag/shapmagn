import torch
import torch.nn as nn
from shapmagn.global_variable import Shape
from shapmagn.metrics.losses import Loss
from shapmagn.utils.obj_factory import obj_factory
from shapmagn.utils.utils import sigmoid_decay
from torch.autograd import grad
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
    where p is the momentum or registration parameter,  v is the velocity guidence solved from gradient flow
    this problem can be either solved via standard conjunction gradient solver or a fast approximate \tilde{v}=Kv



    """
    def __init__(self, opt):
        super(DiscreteFlowOPT, self).__init__()
        self.opt = opt
        self.drift_every_n_iter = self.opt[("drift_every_n_iter",10, "if n is set bigger than -1, then after every n iteration, the current flowed shape is set as the source shape")]
        self.apply_spline_kernel = self.opt[("apply_spline_kernel",True,"smooth the displacement before flow the shape")]
        interpolator_obj = self.opt[("interpolator_obj","point_interpolator.kernel_interpolator(scale=0.1, exp_order=2)", "shape interpolator in multi-scale solver")]
        gauss_smoother_sigma = opt[("gauss_smoother_sigma",0.1, "gauss sigma of the gauss smoother")]
        self.spline_kernel = self.create_spline_kernel(gauss_smoother_sigma)
        self.interp_kernel = obj_factory(interpolator_obj)
        pair_feature_extractor_obj = opt[("pair_feature_extractor_obj", "", "feature extraction function")]
        self.pair_feature_extractor = obj_factory(pair_feature_extractor_obj) if pair_feature_extractor_obj else None
        sim_loss_opt = opt[("sim_loss", {}, "settings for sim_loss_opt")]
        self.sim_loss_fn = Loss(sim_loss_opt)
        self.reg_loss_fn = self.regularization
        self.call_thirdparty_package = False
        self.register_buffer("local_iter", torch.Tensor([0]))
        self.register_buffer("global_iter", torch.Tensor([0]))
        self.print_step = self.opt[('print_step',1,"print every n iteration")]
        self.opt_optim = opt['optim']
        """settings for the optimizer"""
        self.opt_scheduler = opt['scheduler']
        """settings for the scheduler"""
        self.drift_buffer = {}




    def set_loss_fn(self, loss_fn):
        self.sim_loss_fn = loss_fn

    def reset(self):
        self.local_iter = self.local_iter*0



    def flow(self, shape_pair):
        flowed_control_points = shape_pair.flowed_control_points
        toflow_points = shape_pair.toflow.points
        control_points = shape_pair.control_points
        control_weights = shape_pair.control_weights
        flowed_points = self.interp_kernel(toflow_points,control_points, flowed_control_points,control_weights)
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
        assert self.drift_every_n_iter==-1, "the drift mode is not fully tested in multi-scale mode, disabled for now"
        control_points_high = shape_pair_high.get_control_points()
        control_points_low = shape_pair_low.get_control_points()
        control_weights_low = shape_pair_low.control_weights
        reg_param_low = shape_pair_low.reg_param
        reg_param_high = self.interp_kernel(control_points_high, control_points_low, reg_param_low, control_weights_low)
        flowed_control_points_low = shape_pair_low.flowed_control_points.detach()
        interped_control_points_high = self.interp_kernel(control_points_high, control_points_low,
                                                     flowed_control_points_low, control_weights_low)
        shape_pair_high.set_control_points(interped_control_points_high)
        reg_param_high.detach_()
        reg_param_high.requires_grad_()
        shape_pair_high.set_reg_param(reg_param_high)
        if self.drift_every_n_iter>0:
            self.drift_buffer["moving_control_points"] = self.interp_kernel(shape_pair_high.get_control_points(), shape_pair_low.get_control_points(),
                               self.drift_buffer["moving_control_points"], shape_pair_low.control_weights)
        return shape_pair_high


    def drift(self, shape_pair):
        """
        Under drift strategy, the source updates during the optimization,  we introduce a "moving" to record the #current# source
        """
        if self.local_iter % self.drift_every_n_iter==0 and self.local_iter!=0:
            print("drift at the {} th step, the moving (current source) is updated".format(self.local_iter.item()))
            flowed = shape_pair.flowed
            moving = Shape()
            moving.set_data(points=flowed.points.detach().clone(), weights=flowed.weights.detach().clone)
            self.drift_buffer["moving"] = moving
            self.drift_buffer["moving_control_points"] = shape_pair.flowed_control_points.detach().clone()
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


    def create_spline_kernel(self,gauss_smoother_sigma):
        smoother = obj_factory("point_interpolator.kernel_interpolator(scale={}, exp_order=2)".format(gauss_smoother_sigma))
        return smoother






    def smooth_reg_param(self,reg_param, control_points, control_weights):
        if self.apply_spline_kernel:
            return self.spline_kernel(control_points,control_points,reg_param,control_weights)
        else:
            return reg_param

    def standard_spline_forward(self, shape_pair):
        """
        In this forward, a user defined solver should be set
        reg_param here is the displacement
        :param shape_pair:
        :return:
        """
        self.drift(shape_pair)
        if self.drift_every_n_iter == -1 or len(self.drift_buffer) == 0:
            control_points = shape_pair.get_control_points()
        else:
            control_points= self.drift_buffer["moving_control_points"]
        smoothed_reg_param = self.smooth_reg_param(shape_pair.reg_param,control_points, shape_pair.control_weights)
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
            #self.debug(shape_pair.flowed, shape_pair.target)
        loss = sim_loss + reg_loss
        self.local_iter += 1
        self.global_iter +=1
        return loss

    def wasserstein_gradient_flow_guidence(self, flowed, target):
        """
        wassersten gradient flow has a reasonable behavior only when set self.pair_feature_extractor = None
        """

        def update(cur_blur):
            from shapmagn.metrics.losses import GeomDistance
            from torch.autograd import grad
            from copy import deepcopy
            gemloss_setting = deepcopy(self.opt["sim_loss"]["geomloss"])
            gemloss_setting["geom_obj"] = gemloss_setting["geom_obj"].replace("placeholder", str(cur_blur))
            geomloss = GeomDistance(gemloss_setting)
            flowed_points_clone = flowed.points.detach().clone()
            flowed_points_clone.requires_grad_()
            flowed_clone = Shape()
            flowed_clone.set_data_with_refer_to(flowed_points_clone,
                                                flowed)  # shallow copy, only points are cloned, other attr are not
            loss = geomloss(flowed_clone, target)
            print("{} th step, before gradient flow, the ot distance between the flowed and the target is {}".format(
                self.local_iter.item(), loss.item()))
            grad_flowed_points = grad(loss, flowed_points_clone)[0]
            flowed_points_clone = flowed_points_clone - grad_flowed_points / flowed_clone.weights
            flowed_clone.points = flowed_points_clone.detach()
            loss = geomloss(flowed_clone, target)
            print(
                "{} th step, after gradient flow, the ot distance between the gradflowed guided points and the target is {}".format(
                    self.local_iter.item(), loss.item()))
            self.gradflow_guided_buffer["gradflowed"] = flowed_clone

        gradflow_guided_opt = self.opt[("gradflow_guided", {}, "settings for gradflow guidance")]
        self.update_gradflow_every_n_step = gradflow_guided_opt[
            ("update_gradflow_every_n_step", 10, "update the gradflow every # step")]
        gradflow_blur_init = gradflow_guided_opt[
            ("gradflow_blur_init", 0.5, "the inital 'blur' parameter in geomloss setting")]
        update_gradflow_blur_by_raito = gradflow_guided_opt[
            ("update_gradflow_blur_by_raito", 0.5, "the raito that updates the 'blur' parameter in geomloss setting")]
        gradflow_blur_min = gradflow_guided_opt[
            ("gradflow_blur_min", 0.5, "the minium value of the 'blur' parameter in geomloss setting")]
        if self.global_iter % self.update_gradflow_every_n_step == 0 or len(self.gradflow_guided_buffer) == 0:
            n_update = self.global_iter.item() / self.update_gradflow_every_n_step
            cur_blur = max(gradflow_blur_init * (update_gradflow_blur_by_raito ** n_update), gradflow_blur_min)
            update(cur_blur)
        return flowed, self.gradflow_guided_buffer["gradflowed"]

    def gradflow_spline_foward(self, shape_pair):
        self.drift(shape_pair)
        if self.drift_every_n_iter == -1 or len(self.drift_buffer) == 0:
            control_points = shape_pair.get_control_points()
        else:
            control_points = self.drift_buffer["moving_control_points"]
        smoothed_reg_param = self.smooth_reg_param(shape_pair.reg_param, control_points, shape_pair.control_weights)
        flowed_control_points = control_points + smoothed_reg_param
        shape_pair.set_flowed_control_points(flowed_control_points)
        flowed_has_inferred = shape_pair.infer_flowed()
        shape_pair = self.flow(shape_pair) if not flowed_has_inferred else shape_pair
        flowed, target = self.extract_fea(shape_pair.flowed, shape_pair.target)
        flowed, target = self.wasserstein_gradient_flow_guidence(shape_pair.flowed, shape_pair.target)




    def debug(self,flowed, target):
        from shapmagn.utils.visualizer import visualize_point_pair_overlap
        from shapmagn.experiments.datasets.lung.lung_data_analysis import flowed_weight_transform,target_weight_transform
        visualize_point_pair_overlap(flowed.points, target.points,
                                 flowed_weight_transform(flowed.weights,True),
                                 target_weight_transform(target.weights,True),
                                 title1="flowed",title2="target", rgb_on=False)



