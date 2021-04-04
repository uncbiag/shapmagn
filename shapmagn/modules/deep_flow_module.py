import torch
from copy import deepcopy
import torch.nn as nn
from pykeops.torch import LazyTensor
import torch.nn.functional as F
from shapmagn.global_variable import Shape
from shapmagn.utils.obj_factory import obj_factory
from shapmagn.modules.networks.pointnet2.util import PointNetSetAbstraction, PointNetSetUpConv, \
    PointNetFeaturePropogation, FlowEmbedding, index_points
from shapmagn.modules.networks.pointpwcnet_original import multiScaleChamferSmoothCurvature, PointConvSceneFlowPWC8192selfglobalPointConv
from shapmagn.modules.networks.scene_flow import FLOT
from shapmagn.metrics.losses import GeomDistance
from shapmagn.modules.gradient_flow_module import wasserstein_forward_mapping
from shapmagn.modules.networks.flownet3d import FlowNet3D
from shapmagn.modules.networks.pointpwcnet import PointConvSceneFlowPWC
from shapmagn.modules.networks.geo_flow_net import GeoFlowNet
from shapmagn.metrics.losses import CurvatureReg







class DeepFlowNetRegParam(nn.Module):
    def __init__(self,opt):
        super(DeepFlowNetRegParam,self).__init__()
        self.opt = opt
        local_pair_feature_extractor_obj = self.opt[("local_pair_feature_extractor_obj","","function object for local_feature_extractor")]
        self.local_pair_feature_extractor = obj_factory(local_pair_feature_extractor_obj) if len(local_pair_feature_extractor_obj) else self.default_local_pair_feature_extractor
        # todo test the case that the pointfea is initialized by the dataloader
        self.initial_npoints = self.opt[("initial_npoints",4096,"num of initial sampling points")]
        self.input_channel = self.opt[("input_channel",1,"input channel")]
        self.initial_radius = self.opt[("initial_radius",0.001,"initial radius")]
        self.param_factor = self.opt[("param_factor",2,"control the number of factor, #params/param_factor")]
        self.init_deep_feature_extractor()
        self.buffer = {}
        self.iter = 0


    def default_local_pair_feature_extractor(self,cur_source, target):
        cur_source.pointfea = cur_source.weights
        target.pointfea = target.weights
        return cur_source, target

    def init_deep_feature_extractor(self):
        self.flow_predictor = FlowNet3D(input_channel=self.input_channel,initial_radius=self.initial_radius,initial_npoints=self.initial_npoints, param_factor=self.param_factor)


    def deep_flow(self,cur_source, target):
        pc1, pc2, feature1, feature2 = cur_source.points, target.points, cur_source.pointfea, target.pointfea
        pc1, pc2, feature1, feature2 = pc1.transpose(2,1).contiguous(),pc2.transpose(2,1).contiguous(), feature1.transpose(2,1).contiguous(), feature2.transpose(2,1).contiguous()
        nonp_param,additional_param = self.flow_predictor(pc1, pc2, feature1, feature2)
        return nonp_param, None

    def normalize_fea(self):
        pass

    def forward(self,cur_source, target, iter=-1):
        cur_source, target = self.local_pair_feature_extractor(cur_source, target)
        nonp_param,additional_param = self.deep_flow(cur_source, target)
        return nonp_param, additional_param





class DeepGeoNetRegParam(nn.Module):
    def __init__(self,opt):
        super(DeepGeoNetRegParam,self).__init__()
        self.opt = opt
        local_pair_feature_extractor_obj = self.opt[("local_pair_feature_extractor_obj","","function object for local_feature_extractor")]
        self.local_pair_feature_extractor = obj_factory(local_pair_feature_extractor_obj) if len(local_pair_feature_extractor_obj) else self.default_local_pair_feature_extractor
        # todo test the case that the pointfea is initialized by the dataloader
        self.initial_npoints = self.opt[("initial_npoints",4096,"num of initial sampling points")]
        self.input_channel = self.opt[("input_channel",1,"input channel")]
        self.initial_radius = self.opt[("initial_radius",0.001,"initial radius")]
        self.init_deep_feature_extractor()
        self.buffer = {}
        self.iter = 0


    def default_local_pair_feature_extractor(self,cur_source, target):
        cur_source.pointfea = torch.cat([cur_source.points,cur_source.weights],2)
        target.pointfea =torch.cat([target.points,target.weights],2)
        return cur_source, target

    def init_deep_feature_extractor(self):
        self.flow_predictor =GeoFlowNet(input_channel=self.input_channel,initial_radius=self.initial_radius,initial_npoints=self.initial_npoints)

    def deep_flow(self,cur_source, target):
        pc1, pc2, feature1, feature2 = cur_source.points, target.points, cur_source.pointfea, target.pointfea
        nonp_param,additional_param = self.flow_predictor(pc1, pc2, feature1, feature2)
        return nonp_param, None

    def normalize_fea(self):
        pass

    def forward(self,cur_source, target, iter=-1):
        cur_source, target = self.local_pair_feature_extractor(cur_source, target)
        nonp_param,additional_param = self.deep_flow(cur_source, target)
        return nonp_param, additional_param





class PointConvSceneFlowPWCRegParam(nn.Module):
    def __init__(self, opt):
        super(PointConvSceneFlowPWCRegParam, self).__init__()
        self.opt = opt
        local_pair_feature_extractor_obj = self.opt[("local_pair_feature_extractor_obj","","function object for local_feature_extractor")]
        self.local_pair_feature_extractor = obj_factory(local_pair_feature_extractor_obj) if len(local_pair_feature_extractor_obj) else self.default_local_pair_feature_extractor
        # todo test the case that the pointfea is initialized by the dataloader
        self.input_channel = self.opt[("input_channel",1,"input channel")]
        self.initial_npoints = self.opt[("initial_npoints",2048,"num of initial sampling points")]
        self.initial_radius = self.opt[("initial_radius",0.001,"initial radius or the resolution of the point cloud")]
        self.delploy_original_model = self.opt[("delploy_original_model",False,"delploy the original model in pointpwcnet paper")]
        self.init_deep_feature_extractor()
        self.buffer = {}
        self.iter = 0

    def default_local_pair_feature_extractor(self, cur_source, target):
        cur_source.pointfea = cur_source.points.clone()#torch.cat([cur_source.points, cur_source.weights], 2)
        target.pointfea = target.points.clone() #torch.cat([target.points, target.weights], 2)
        return cur_source, target

    def init_deep_feature_extractor(self):
        self.load_pretrained_model = self.opt[("load_pretrained_model",False,"load_pretrained_model")]
        self.pretrained_model_path = self.opt[("pretrained_model_path","","path of pretrained model")]
        if not self.delploy_original_model:
            self.flow_predictor = PointConvSceneFlowPWC(input_channel=self.input_channel, initial_radius=self.initial_radius, initial_npoints=self.initial_npoints)
        else:
            self.flow_predictor = PointConvSceneFlowPWC8192selfglobalPointConv(input_channel=self.input_channel,  initial_npoints=self.initial_npoints)
        if self.load_pretrained_model:
            checkpoint = torch.load(self.pretrained_model_path, map_location='cpu')
            self.flow_predictor.load_state_dict(checkpoint)
    def deep_flow(self, cur_source, target):
        pc1, pc2, feature1, feature2 = cur_source.points, target.points, cur_source.pointfea, target.pointfea
        nonp_param,additional_param = self.flow_predictor(pc1, pc2, feature1, feature2)
        return nonp_param, additional_param

    def forward(self, cur_source, target, iter=-1):
        cur_source, target = self.local_pair_feature_extractor(cur_source, target)
        nonp_param, additional_param = self.deep_flow(cur_source, target)
        return nonp_param, additional_param








class FLOTRegParam(nn.Module):
    def __init__(self, opt):
        super(FLOTRegParam, self).__init__()
        self.opt = opt
        self.input_channel = self.opt[("input_channel",1,"input channel")]
        self.nb_iter = self.opt[("nb_iter",1,"# iter for solving the ot problem")]
        self.init_deep_feature_extractor()
        self.buffer = {}
        self.iter = 0

    def init_deep_feature_extractor(self):
        self.flow_predictor = FLOT(nb_iter=self.nb_iter, initial_channel=self.input_channel)

    def deep_flow(self, cur_source, target):
        pc1, pc2 = cur_source.points, target.points
        nonp_param,additional_param = self.flow_predictor(pc1, pc2)
        return nonp_param, additional_param

    def forward(self, cur_source, target, iter=-1):
        nonp_param, additional_param = self.deep_flow(cur_source, target)
        return nonp_param, additional_param





class FlowModel(nn.Module):
    def __init__(self, opt):
        super(FlowModel, self).__init__()
        self.opt = opt
        self.model_type = opt[("model_type","spline","model type 'spline'/'lddmm'")]
        if self.model_type=="spline":
            self.init_spline(opt["spline"])
            self.flow_model = self.spline_forward
            self.regularization = self.spline_reg
        elif self.model_type=="lddmm":
            self.init_lddmm(opt["lddmm"])
            self.flow_model = self.lddmm_forward
            self.regularization = self.lddmm_reg
        elif self.model_type == "disp":
            self.flow_model = self.disp_forward
            self.regularization = self.disp_reg
        self.init_interp(opt[("interp",{},"settings for interpolation mode")])


    def init_interp(self,opt_interp):
        if self.model_type== "disp" or self.model_type=="spline":
            interp_kernel_obj = opt_interp[(
            "interp_kernel_obj", "point_interpolator.NadWatIsoSpline(exp_order=2,kernel_scale=0.05)",
            "shape interpolator")]
            self.interp_kernel = obj_factory(interp_kernel_obj)

    def interp(self,toflow_points, control_points, control_value, control_weights, reg_param=None):
        if self.model_type== "disp" or self.model_type=="spline":
            return self.interp_kernel(toflow_points, control_points, control_value, control_weights)
        elif self.model_type=="lddmm":
            self.lddmm_module.set_mode("flow")
            momentum = reg_param
            _, _, flowed_points = self.integrator.solve((momentum, control_points, toflow_points))
            return flowed_points
        else:
            raise NotImplemented


    def disp_forward(self,toflow, reg_param):
        points, weights = toflow.points, toflow.weights
        flowed_points = points + reg_param
        return flowed_points, None

    def disp_reg(self,reg_param, reg_additional_input=None):
        dist = reg_param ** 2
        dist = dist.sum(2).mean(1)
        return dist


    def init_spline(self,spline_opt):
        spline_kernel_obj = spline_opt[("spline_kernel_obj","point_interpolator.NadWatIsoSpline(exp_order=2,kernel_scale=0.05)", "shape interpolator")]
        self.spline_kernel = obj_factory(spline_kernel_obj)


    def spline_forward(self,toflow, reg_param):
        """
        input the reg_param and output the displacement
        :param reg_param:
        :return:
        """
        points, weights = toflow.points, toflow.weights
        sm_disp = self.spline_kernel(points,points,reg_param,weights) #todo check if detach the points
        flowed_points = points + sm_disp
        return flowed_points, sm_disp

    def spline_reg(self,disp, sm_disp):
        dist = sm_disp * disp
        dist = dist.sum(2).mean(1)
        return dist

    def init_lddmm(self, lddmm_opt):
        from shapmagn.modules.ode_int import ODEBlock
        from shapmagn.modules.lddmm_module import LDDMMHamilton, LDDMMVariational
        self.module_type = lddmm_opt[("module", "hamiltonian", "lddmm module type: hamiltonian or variational")]
        assert self.module_type in ["hamiltonian", "variational"]
        self.lddmm_module = LDDMMHamilton(lddmm_opt[("hamiltonian", {}, "settings for hamiltonian")]) \
            if self.module_type == 'hamiltonian' else LDDMMVariational(
            lddmm_opt[("variational", {}, "settings for variational")])
        self.lddmm_module.set_mode("shooting")
        self.lddmm_kernel = self.lddmm_module.kernel
        self.integrator_opt = lddmm_opt[("integrator", {}, "settings for integrator")]
        self.integrator = ODEBlock(self.integrator_opt)
        self.integrator.set_func(self.lddmm_module)

    def lddmm_forward(self, toflow, reg_param):
        toflow_points = toflow.points
        momentum = reg_param
        momentum = momentum.clamp(-0.5, 0.5)
        _, flowed_points = self.integrator.solve((momentum, toflow_points))
        return flowed_points, toflow_points

    def lddmm_reg(self,momentum, toflow_points):
        momentum = momentum.clamp(-0.5, 0.5)
        dist = momentum * self.lddmm_kernel(toflow_points, toflow_points, momentum) #todo check if detach the points
        dist = dist.sum(2).mean(1)
        return dist


    def forward(self, source, reg_param):
        flowed_points, reg_additional_input = self.flow_model(source,reg_param)
        reg_loss = self.regularization(reg_param, reg_additional_input)
        flowed = Shape().set_data_with_refer_to(flowed_points,source)
        return flowed , reg_loss




class DeepFlowLoss(nn.Module):
    def __init__(self, opt):
        super(DeepFlowLoss,self).__init__()
        self.opt = opt
        self.loss_type = self.opt[("loss_type", "disp_l2","disp_l2 / ot_loss")]
        self.include_curv_constrain =self.opt[("include_curv_constrain", True, "add constrain on the curvature")]
        self.curv_factor =self.opt[("curv_factor", 1.0, "curv_factor")]



    def disp_l2(self, flowed, target):
        """
        the gt correspondence should be known and the number of points should be the same
        :param flowed:
        :param target:
        :return:
        """
        fp = flowed.points
        tp = target.points
        fw = flowed.weights
        l2_loss = (((fp-tp)**2).sum(2,keepdim=True) * fw).sum(1) #todo test
        return l2_loss[...,0] # remove the last 1 dim


    def ot_distance(self,flowed, target):
        """
        the gt correspondence can be unknown, the point number can be inconsistent
        :param flowed: shape,
        :param target: shape
        :return:
        """
        geomloss_setting = deepcopy(self.opt["geomloss"])
        geomloss_setting["attr"] = "points"
        self.geom_loss = GeomDistance(geomloss_setting)
        return self.geom_loss(flowed, target)


    def geo_distance(self,flowed,target):
        """
        compute the local feature of  the flowed and the target, compare there l2 difference
        :param flowed:
        :param target:
        :return:
        """
        local_pair_feature_extractor_obj = self.opt[
            ("local_pair_feature_extractor_obj", "", "function object for local_feature_extractor")]
        local_pair_feature_extractor = obj_factory(local_pair_feature_extractor_obj) if len(
            local_pair_feature_extractor_obj) else None
        flowed, target = local_pair_feature_extractor(flowed, target)
        fp = flowed.pointfea
        tp = target.pointfea
        fw = flowed.weights
        l2_loss = (((fp - tp) ** 2).sum(2, keepdim=True) * fw).sum(1)  # todo test
        return l2_loss[..., 0]


    def curvature_diff(self,source, flowed, target):
        curloss_setting = deepcopy(self.opt["curloss"])
        curloss = GeomDistance(curloss_setting)
        curvature_loss = curloss(source, flowed, target)
        return curvature_loss



    def forward(self,flowed, target, additional_param=None, has_gt=True):
        curv_reg = self.curv_factor* self.curvature_diff(additional_param["source"], flowed,
                                       target) if self.include_curv_constrain else 0
        if not has_gt:
            return self.ot_distance(flowed, target) + curv_reg
        if self.loss_type == "disp_l2":
            return self.disp_l2(flowed, target) + curv_reg







class PWCLoss(nn.Module):
    def __init__(self,opt):
        super(PWCLoss,self).__init__()
        self.opt = opt
        self.multi_scale_weight = opt[("multi_scale_weight",[0.02, 0.04, 0.08, 0.16],"weight for each scale")]
        self.loss_type = opt[("loss_type", "multi_scale","multi_scale / chamfer_self")]
        self.use_self_supervised_loss = opt[("use_self_supervised_loss", False,"use_self_supervised_loss")]


    def multiScaleLoss(self,flowed, target, additional_param):
        weights = flowed.weights
        alpha = self.multi_scale_weight
        floweds, fps_idxs = additional_param["floweds"], additional_param["fps_pc1_idxs"]#todo to be fixed
        num_scale = len(floweds)
        offset = len(fps_idxs) - num_scale + 1
        num_scale = len(floweds)
        #generate GT list and mask1s
        gt = [target.points]
        w = [weights]
        for i in range(len(fps_idxs)):
            fps_idx = fps_idxs[i]
            sub_gt_flow = index_points(gt[-1], fps_idx)
            sub_w = index_points(w[-1],fps_idx)
            gt.append(sub_gt_flow)
            w.append(sub_w/sub_w.sum(1,keepdim=True))
        total_loss = 0
        for i in range(num_scale):
            diff_flow = floweds[i] - gt[i+offset]
            total_loss += alpha[i] * ((diff_flow**2).sum(dim = 2,keepdim=True)*w[i]).sum(1)
        return total_loss[...,0]


    def chamfer_self_loss(self,flowed, target, additional_param):
        pred_flows, pc1, pc2 = additional_param["flows"],additional_param["pc1"], additional_param["pc2"]
        total_loss, chamfer_loss, curvature_loss, smoothness_loss = multiScaleChamferSmoothCurvature(pc1, pc2, pred_flows)
        return total_loss



    def forward(self,flowed, target, additional_param=None, has_gt=True):
        if self.use_self_supervised_loss:
            return self.chamfer_self_loss(flowed, target, additional_param)
        elif has_gt:
            return self.multiScaleLoss(flowed, target, additional_param)
        else:
            raise NotImplemented




if __name__ == "__main__":
    import torch
    from pykeops.torch import LazyTensor
    def deep_corres(points, weights, pointfea):
        x_i = LazyTensor(points[:, :, None])  # BxNx1xD
        x_j = LazyTensor(points[:, None])  # Bx1xNxD
        fea_i = LazyTensor(pointfea[:, :, None])  # BxNx1xd
        fea_j = LazyTensor(pointfea[:, None])  # Bx1xNxd
        weight = LazyTensor(weights[:, None])  # Bx1xNx1
        point_dist2 = x_i.sqdist(x_j)  # BxNxNx1
        fea_dist2 = fea_i.sqdist(fea_j)  # BxNxNx1
        point_dist2 = point_dist2 - point_dist2.max()  # BxNxN
        fea_dist2 = fea_dist2 - fea_dist2.max()  # BxNxN
        point_logsoftmax = point_dist2 - LazyTensor(point_dist2.logsumexp(dim=2)[...,None])  # BxNxN
        fea_logsoftmax = fea_dist2 - LazyTensor(fea_dist2.logsumexp(dim=2)[...,None])  # BxNxN
        loss = ((point_logsoftmax.exp() - fea_logsoftmax.exp()) * weight) ** 2
        return loss.sum(2).sum(1)

    points = torch.rand(2,1000,3)
    fea = torch.rand(2,1000,5)
    weights = torch.rand(2,1000,1)
    deep_corres(points, weights, fea)
