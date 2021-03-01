import torch
from copy import deepcopy
import torch.nn as nn
from pykeops.torch import LazyTensor
import torch.nn.functional as F
from shapmagn.global_variable import Shape
from shapmagn.utils.obj_factory import obj_factory
from shapmagn.modules.networks.pointnet2.util import PointNetSetAbstraction, PointNetSetUpConv, PointNetFeaturePropogation,FlowEmbedding
from shapmagn.metrics.losses import GeomDistance
from shapmagn.modules.gradient_flow_module import wasserstein_forward_mapping



class DeepRegParm(nn.Module):
    def __init__(self,opt):
        super(DeepRegParm,self).__init__()
        self.opt = opt
        local_pair_feature_extractor_obj = self.opt[("local_pair_feature_extractor_obj","","function object for local_feature_extractor")]
        self.local_pair_feature_extractor = obj_factory(local_pair_feature_extractor_obj) if len(local_pair_feature_extractor_obj) else self.default_local_pair_feature_extractor
        # todo test the case that the pointfea is initialized by the dataloader
        self.input_channel = self.opt[("input_channel",1,"input channel")]
        self.initial_radius = self.opt[("initial_radius",0.001,"initial radius")]
        self.init_deep_feature_extractor()
        self.buffer = {}
        self.iter = 0


    def default_local_pair_feature_extractor(self,cur_source, target):
        cur_source.pointfea = cur_source.weights
        target.pointfea = target.weights
        return cur_source, target




    def init_deep_feature_extractor(self):
        self.sa1 = PointNetSetAbstraction(npoint=4096, radius=20*self.initial_radius, nsample=16, in_channel=self.input_channel, mlp=[16, 16, 32],
                                          group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=1024, radius=40*self.initial_radius, nsample=16, in_channel=32, mlp=[32, 32, 64],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=256, radius=80*self.initial_radius, nsample=8, in_channel=64, mlp=[64, 64, 128],
                                          group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=64, radius=160*self.initial_radius, nsample=8, in_channel=128, mlp=[128, 128, 256],
                                          group_all=False)

        self.fe_layer = FlowEmbedding(radius=500*self.initial_radius, nsample=64, in_channel=64, mlp=[64, 64, 64], pooling='max',
                                      corr_func='concat')

        self.su1 = PointNetSetUpConv(nsample=8, radius=96*self.initial_radius, f1_channel=128, f2_channel=256, mlp=[], mlp2=[128, 128])
        self.su2 = PointNetSetUpConv(nsample=8, radius=48*self.initial_radius, f1_channel=64 + 64, f2_channel=128, mlp=[64, 64, 128],
                                     mlp2=[128])
        self.su3 = PointNetSetUpConv(nsample=8, radius=24*self.initial_radius, f1_channel=32, f2_channel=128, mlp=[64, 64, 128],
                                     mlp2=[128])
        self.fp = PointNetFeaturePropogation(in_channel=128 + self.input_channel, mlp=[128, 128])

        self.conv1 = nn.Conv1d(128, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 3, kernel_size=1, bias=True)


    def deep_flow(self,cur_source, target):
        pc1, pc2, feature1, feature2 = cur_source.points, target.points, cur_source.pointfea, target.pointfea
        pc1, pc2, feature1, feature2 = pc1.transpose(2,1).contiguous(),pc2.transpose(2,1).contiguous(), feature1.transpose(2,1).contiguous(), feature2.transpose(2,1).contiguous()

        l1_pc1, l1_feature1 = self.sa1(pc1, feature1)
        l2_pc1, l2_feature1 = self.sa2(l1_pc1, l1_feature1)

        l1_pc2, l1_feature2 = self.sa1(pc2, feature2)
        l2_pc2, l2_feature2 = self.sa2(l1_pc2, l1_feature2)

        _, l2_feature1_new = self.fe_layer(l2_pc1, l2_pc2, l2_feature1, l2_feature2)

        l3_pc1, l3_feature1 = self.sa3(l2_pc1, l2_feature1_new)
        l4_pc1, l4_feature1 = self.sa4(l3_pc1, l3_feature1)

        l3_fnew1 = self.su1(l3_pc1, l4_pc1, l3_feature1, l4_feature1)
        l2_fnew1 = self.su2(l2_pc1, l3_pc1, torch.cat([l2_feature1, l2_feature1_new], dim=1), l3_fnew1)
        l1_fnew1 = self.su3(l1_pc1, l2_pc1, l1_feature1, l2_fnew1)
        l0_fnew1 = self.fp(pc1, l1_pc1, feature1, l1_fnew1)

        x = F.relu(self.bn1(self.conv1(l0_fnew1)))
        nonp_param = self.conv2(x)
        nonp_param = nonp_param.transpose(2,1).contiguous()

        return nonp_param

    def normalize_fea(self):
        pass

    def forward(self,cur_source, target, iter=-1):
        cur_source, target = self.local_pair_feature_extractor(cur_source, target)
        nonp_param = self.deep_flow(cur_source, target)
        return nonp_param

class FlowModel(nn.Module):
    def __init__(self, opt):
        super(FlowModel, self).__init__()
        self.opt = opt
        self.model_type = opt[("model_type","spline","model type 'spline'/'lddmm'")]
        self.n_step = opt[("n_step",1,"number of iteration step")]
        if self.model_type=="spline":
            self.init_spline(opt["spline"])
            self.flow_model = self.spline_forward
            self.reguralization = self.spline_reg
        else:
            self.init_lddmm(opt["lddmm"])
            self.flow_model = self.lddmm_forward
            self.reguralization = self.lddmm_reg



    def init_spline(self,spline_opt):
        spline_kernel_obj = spline_opt[("spline_kernel_obj","point_interpolator.NadWatIsoSpline(exp_order=2,kernel_scale=0.05)", "shape interpolator in multi-scale solver")]
        self.spline_kernel = obj_factory(spline_kernel_obj)


    def spline_forward(self,toflow, reg_param):
        """
        input the reg_param and output the displacement
        :param reg_param:
        :return:
        """
        points, weights = toflow.points, toflow.weights
        sm_disp = self.spline_kernel(points,points,reg_param,weights)
        flowed_points = points + sm_disp
        return flowed_points, sm_disp

    def spline_reg(self,disp, sm_disp):
        dist = sm_disp * disp
        dist = dist.mean(2).mean(1)
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
        dist = momentum * self.lddmm_kernel(toflow_points, toflow_points, momentum)
        dist = dist.mean(2).mean(1)
        return dist


    def forward(self, source, reg_param):
        flowed_points, reg_additional_input = self.flow_model(source,reg_param)
        reg_loss = self.reguralization(reg_param, reg_additional_input)
        flowed = Shape().set_data_with_refer_to(flowed_points,source)
        return flowed , reg_loss



class DeepFlowLoss(nn.Module):
    def __init__(self, opt):
        super(DeepFlowLoss,self).__init__()
        self.opt = opt
        self.loss_type = self.opt[("loss_type", "disp_l2","disp_l2 / ot_loss")]
        geomloss_setting = deepcopy(self.opt["geomloss"])
        geomloss_setting["attr"] = "points"
        self.geom_loss = GeomDistance(geomloss_setting)
        self.buffer = {"gt_one_hot":None, "gt_plan":None}



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

        l2_loss = (((fp-tp)**2).mean(2,keepdim=True) * fw).sum() #todo test
        return l2_loss


    def ot_distance(self,flowed, target):
        """
        the gt correspondence can be unknown, the point number can be inconsistent
        :param flowed: shape,
        :param target: shape
        :return:
        """
        return self.geom_loss(flowed, target)


    def forward(self,flowed, target):
        if self.loss_type == "disp_l2":
            return self.disp_l2(flowed, target)
        elif self.loss_type == "ot_corres":
            return self.ot_distance(flowed, target)





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
