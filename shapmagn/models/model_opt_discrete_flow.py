import torch
import torch.nn as nn
from shapmagn.global_variable import Shape
from shapmagn.metrics.losses import Loss
from shapmagn.utils.obj_factory import obj_factory
from shapmagn.utils.utils import sigmoid_decay
from shapmagn.modules.optimizer import optimizer_builder
from shapmagn.modules.scheduler import scheduler_builder
from torch.autograd import grad
class DiscreteFlowOPT(nn.Module):
    def __init__(self, opt):
        super(DiscreteFlowOPT, self).__init__()
        self.opt = opt
        self.drift_every_n_iter = self.opt[("drift_every_n_iter",10, "if n is set bigger than -1, then after every n iteration, the current flowed shape is set as the source shape")]
        self.smooth_displacement = self.opt[("smooth_displacement",True,"smooth the displacement before flow the shape")]
        interpolator_obj = self.opt[("interpolator_obj","point_interpolator.kernel_interpolator(scale=0.1, exp_order=2)", "shape interpolator in multi-scale solver")]
        gauss_smoother_sigma = opt[("gauss_smoother_sigma",0.1, "gauss sigma of the gauss smoother")]
        self.gauss_kernel = self.create_gauss_smoother(gauss_smoother_sigma)
        self.interp_kernel = obj_factory(interpolator_obj)
        feature_extractor_obj = opt[("feature_extractor_obj", "", "feature extraction function")]
        self.feature_extractor = obj_factory(feature_extractor_obj) if feature_extractor_obj else None
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
        Under drift strategy, the source update during the optimization,  we introduce a "moving" to record the #current# source
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
        if not self.feature_extractor:
            return self.extract_point_fea(flowed, target)
        else:
            return self.feature_extractor(flowed,target)


    def create_gauss_smoother(self,gauss_smoother_sigma):
        smoother = obj_factory("point_interpolator.kernel_interpolator(scale={}, exp_order=2)".format(gauss_smoother_sigma))
        return smoother

    def smooth_reg_param(self,reg_param, control_points, control_weights):
        if self.smooth_displacement:
            return self.gauss_kernel(control_points,control_points,reg_param,control_weights)
        else:
            return reg_param

    def forward(self, shape_pair):
        """
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



    def debug(self,flowed, target):
        from shapmagn.utils.visualizer import visualize_point_pair_overlap
        from shapmagn.experiments.datasets.lung.lung_data_analysis import flowed_weight_transform,target_weight_transform
        visualize_point_pair_overlap(flowed.points, target.points,
                                 flowed_weight_transform(flowed.weights,True),
                                 target_weight_transform(target.weights,True),
                                 title1="flowed",title2="target", rgb_on=False)



