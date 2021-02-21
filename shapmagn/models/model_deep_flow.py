from copy import deepcopy
import torch
import torch.nn as nn
from shapmagn.global_variable import Shape
from shapmagn.metrics.losses import Loss
from shapmagn.utils.obj_factory import obj_factory,partial_obj_factory
from shapmagn.utils.utils import sigmoid_decay
class DeepDiscreteFlow(nn.Module):
    """
    flow the source via n step, in each step with the #current# source X get updated, the target Y is fixed
    In this class, we provide two approaches to solve this task:



    """
    def __init__(self, opt):
        super(DeepDiscreteFlow, self).__init__()
        self.opt = opt
        self.drift_every_n_iter = self.opt[("drift_every_n_iter",10, "if n is set bigger than -1, then after every n iteration, the current flowed shape is set as the source shape")]
        spline_kernel_obj = self.opt[("spline_kernel_obj","point_interpolator.NadWatIsoSpline(exp_order=2,kernel_scale=0.05)", "shape interpolator in multi-scale solver")]
        self.spline_kernel= obj_factory(spline_kernel_obj)
        pair_feature_extractor_obj = self.opt[("pair_feature_extractor_obj", "", "feature extraction function")]
        interp_kernel_obj = self.opt[(
        "interp_kernel_obj", "point_interpolator.NadWatIsoSpline(exp_order=2,kernel_scale=0.05)",
        "kernel for multi-scale interpolation")]
        self.interp_kernel = obj_factory(interp_kernel_obj)
        self.pair_feature_extractor = obj_factory(pair_feature_extractor_obj) if pair_feature_extractor_obj else None
        self.use_aniso_kernel = self.opt[("use_aniso_kernel",False,"use anisotropic kernel")]
        sim_loss_opt = opt[("sim_loss", {}, "settings for sim_loss_opt")]
        self.sim_loss_fn = Loss(sim_loss_opt)
        self.reg_loss_fn = self.regularization
        self.register_buffer("local_iter", torch.Tensor([0]))
        self.print_step = self.opt[('print_step',1,"print every n iteration")]
        self.drift_buffer = {}
        if self.gradient_flow_mode:
            print("in gradient flow mode, points drift every iteration")
            self.drift_every_n_iter =1


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
        self.spline_kernel.set_flow(True)
        interped_control_points_disp = self.spline_kernel(toflow_points, control_points,
                                                               shape_pair.reg_param, control_weights)
        self.spline_kernel.set_flow(False)
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



    def standard_spline_forward(self, shape_pair):
        """
        In this forward, the solver should be built either via build_multi_scale_solver / build_single_scale_custom_solver
        reg_param here is the momentum p before kernel operation K,  \tilde{v} = Kp
        :param shape_pair:
        :return:
        """
        if self.drift_every_n_iter == -1:
            control_points = shape_pair.get_control_points()
            moving = shape_pair.source
        else:
            self.drift(shape_pair)
            control_points, moving= self.drift_buffer["moving_control_points"], self.drift_buffer["moving"]
        # todo check the behavior of spline kernel with shape_pair.control points as input
        smoothed_reg_param = self.spline_kernel(control_points,control_points,shape_pair.reg_param,
                                                shape_pair.control_weights)
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
            #self.debug(shape_pair)
        loss = sim_loss + reg_loss
        self.local_iter += 1
        self.global_iter +=1
        return loss



    def forward(self, shape_pair):
        if self.gradient_flow_mode:
            return self.gradflow_spline_foward(shape_pair)
        else:
            return self.standard_spline_forward(shape_pair)




