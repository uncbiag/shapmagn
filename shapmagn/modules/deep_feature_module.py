from copy import deepcopy
import torch
import torch.nn as nn
from pykeops.torch import LazyTensor
import torch.nn.functional as F
from shapmagn.global_variable import Shape
from shapmagn.utils.obj_factory import obj_factory
from shapmagn.modules.networks.pointnet2.util import PointNetSetAbstraction, PointNetSetUpConv, PointNetFeaturePropogation
from shapmagn.metrics.losses import GeomDistance
from shapmagn.modules.gradient_flow_module import wasserstein_forward_mapping



class PointNet2FeaExtractor(nn.Module):
    def __init__(self,opt):
        super(PointNet2FeaExtractor,self).__init__()
        self.opt = opt
        self.local_pair_feature_extractor = obj_factory(self.opt[("local_pair_feature_extractor_obj","","function object for local_feature_extractor")])
        self.input_channel = self.opt[("input_channel",10,"input channel")]
        self.initial_radius = self.opt[("initial_radius",0.001,"initial radius")]
        self.include_pos_in_final_feature = self.opt[("include_pos_in_final_feature", True, "include_pos")]
        self.include_pos_info_network = self.opt[("include_pos_info_network",True,"include_pos")]
        self.use_complex_network = self.opt[("use_complex_network",False,"use a complex pointnet to extract feature ")]
        if not self.use_complex_network:
            self.init_simple_deep_feature_extractor()
            self.deep_pair_feature_extractor = self.simple_deep_pair_feature_extractor
        else:
            self.init_complex_deep_feature_extractor()
            self.deep_pair_feature_extractor = self.complex_deep_pair_feature_extractor
        self.buffer = {}
        self.iter = 0


    def init_simple_deep_feature_extractor(self):
        self.sa1 = PointNetSetAbstraction(npoint=4096, radius=self.initial_radius*20, nsample=32, in_channel=self.input_channel, mlp=[16,32],
                                          group_all=False, include_xyz=self.include_pos_info_network)
        self.sa2 = PointNetSetAbstraction(npoint=1024, radius=self.initial_radius*40, nsample=32, in_channel=32, mlp=[32, 64],
                                          group_all=False, include_xyz=self.include_pos_info_network)
        self.su1 = PointNetSetUpConv(nsample=8, radius=self.initial_radius*24, f1_channel=64, f2_channel=64, mlp=[32], mlp2=[32])
        self.fp = PointNetFeaturePropogation(in_channel=32+self.input_channel, mlp=[32])
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 12, kernel_size=1, bias=True)


    def simple_deep_pair_feature_extractor(self,cur_source, target):
        pc1, pc2, feature1, feature2 = cur_source.points, target.points, cur_source.pointfea, target.pointfea
        pc1, pc2, feature1, feature2 = pc1.transpose(2,1).contiguous(),pc2.transpose(2,1).contiguous(), feature1.transpose(2,1).contiguous(), feature2.transpose(2,1).contiguous()

        l1_pc1, l1_feature1 = self.sa1(pc1, feature1)
        l2_pc1, l2_feature1 = self.sa2(l1_pc1, l1_feature1)
        l1_fnew1 = self.su1(l1_pc1, l2_pc1,l1_feature1, l2_feature1)
        l0_fnew1 = self.fp(pc1, l1_pc1, feature1, l1_fnew1)
        x = F.relu(self.bn1(l0_fnew1))
        sf = self.conv2(x)


        l1_pc2, l1_feature2 = self.sa1(pc2, feature2)
        l2_pc2, l2_feature2 = self.sa2(l1_pc2, l1_feature2)
        l1_fnew2 = self.su1(l1_pc2, l2_pc2, l1_feature2, l2_feature2)
        l0_fnew2 = self.fp(pc2, l1_pc2, feature2, l1_fnew2)
        x = F.relu(self.bn1(l0_fnew2))
        tf = self.conv2(x)

        # sf = (sf-sf.mean(2,keepdim=True))/(sf.std(2,keepdim=True)+1e-7)
        # tf = (tf-tf.mean(2,keepdim=True))/(tf.std(2,keepdim=True)+1e-7)

        sf, tf = sf.transpose(2,1).contiguous(), tf.transpose(2,1).contiguous()


        if self.include_pos_in_final_feature:
            sf = torch.cat([cur_source.points,sf],2)
            tf = torch.cat([target.points,tf],2)

        new_cur_source = Shape().set_data(points=cur_source.points,weights=cur_source.weights, pointfea=sf)
        new_target = Shape().set_data(points=target.points,weights=target.weights, pointfea=tf)
        return new_cur_source, new_target

    def init_complex_deep_feature_extractor(self):
        self.sa1 = PointNetSetAbstraction(npoint=4096, radius=20 * self.initial_radius, nsample=16,
                                          in_channel=self.input_channel, mlp=[16, 16, 32],
                                          group_all=False, include_xyz=self.include_pos_info_network)
        self.sa2 = PointNetSetAbstraction(npoint=1024, radius=40 * self.initial_radius, nsample=16, in_channel=32,
                                          mlp=[32, 32, 64],
                                          group_all=False, include_xyz=self.include_pos_info_network)
        self.sa3 = PointNetSetAbstraction(npoint=256, radius=80 * self.initial_radius, nsample=8, in_channel=64,
                                          mlp=[64, 64, 128],
                                          group_all=False, include_xyz=self.include_pos_info_network)
        self.sa4 = PointNetSetAbstraction(npoint=64, radius=160 * self.initial_radius, nsample=8, in_channel=128,
                                          mlp=[128, 128, 256],
                                          group_all=False, include_xyz=self.include_pos_info_network)

        self.su1 = PointNetSetUpConv(nsample=8, radius=96 * self.initial_radius, f1_channel=128, f2_channel=256, mlp=[],
                                     mlp2=[128, 128])
        self.su2 = PointNetSetUpConv(nsample=8, radius=48 * self.initial_radius, f1_channel=64, f2_channel=128,
                                     mlp=[64, 64, 128],
                                     mlp2=[128])
        self.su3 = PointNetSetUpConv(nsample=8, radius=24 * self.initial_radius, f1_channel=32, f2_channel=128,
                                     mlp=[64, 64, 128],
                                     mlp2=[128])
        self.fp = PointNetFeaturePropogation(in_channel=128 + self.input_channel, mlp=[128, 128])

        self.conv1 = nn.Conv1d(128, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 12, kernel_size=1, bias=True)

    def complex_deep_pair_feature_extractor(self, cur_source, target):
        pc1, pc2, feature1, feature2 = cur_source.points, target.points, cur_source.pointfea, target.pointfea
        pc1, pc2, feature1, feature2 = pc1.transpose(2, 1).contiguous(), pc2.transpose(2, 1).contiguous(), feature1.transpose(
            2, 1).contiguous(), feature2.transpose(2, 1).contiguous()

        l1_pc1, l1_feature1 = self.sa1(pc1, feature1)
        l2_pc1, l2_feature1 = self.sa2(l1_pc1, l1_feature1)
        l3_pc1, l3_feature1 = self.sa3(l2_pc1, l2_feature1)
        l4_pc1, l4_feature1 = self.sa4(l3_pc1, l3_feature1)
        l3_fnew1 = self.su1(l3_pc1, l4_pc1, l3_feature1, l4_feature1)
        l2_fnew1 = self.su2(l2_pc1, l3_pc1, l2_feature1, l3_fnew1)
        l1_fnew1 = self.su3(l1_pc1, l2_pc1, l1_feature1, l2_fnew1)
        l0_fnew1 = self.fp(pc1, l1_pc1, feature1, l1_fnew1)
        x = F.relu(self.bn1(self.conv1(l0_fnew1)))
        sf = self.conv2(x)
        sf = sf.transpose(2, 1).contiguous()


        l1_pc2, l1_feature2 = self.sa1(pc2, feature2)
        l2_pc2, l2_feature2 = self.sa2(l1_pc2, l1_feature2)
        l3_pc2, l3_feature2 = self.sa3(l2_pc2, l2_feature2)
        l4_pc2, l4_feature2 = self.sa4(l3_pc2, l3_feature2)
        l3_fnew2 = self.su1(l3_pc2, l4_pc2, l3_feature2, l4_feature2)
        l2_fnew2 = self.su2(l2_pc2, l3_pc2, l2_feature2, l3_fnew2)
        l1_fnew2 = self.su3(l1_pc2, l2_pc2, l1_feature2, l2_fnew2)
        l0_fnew2 = self.fp(pc2, l1_pc2, feature2, l1_fnew2)
        x = F.relu(self.bn1(self.conv1(l0_fnew2)))
        tf = self.conv2(x)
        tf = tf.transpose(2, 1).contiguous()

        if self.include_pos_in_final_feature:
            sf = torch.cat([cur_source.points, sf], 2)
            tf = torch.cat([target.points, tf], 2)

        new_cur_source = Shape().set_data(points=cur_source.points, weights=cur_source.weights, pointfea=sf)
        new_target = Shape().set_data(points=target.points, weights=target.weights, pointfea=tf)
        return new_cur_source, new_target

    def normalize_fea(self):
        pass

    def __call__(self,cur_source, target, iter=-1):
        cur_source, target = self.local_pair_feature_extractor(cur_source, target)
        cur_source, target = self.deep_pair_feature_extractor(cur_source, target)

        return cur_source, target




class DeepFeatureLoss(nn.Module):
    def __init__(self, opt):
        super(DeepFeatureLoss,self).__init__()
        self.opt = opt
        self.loss_type = self.opt[("loss_type", "ot_map","naive_corres / ot_corres /ot_ce/ot_soften_ce/ ot_distance/ot_map")]
        self.geomloss_setting = deepcopy(self.opt["geomloss"])
        self.geomloss_setting.print_settings_off()
        self.geom_loss = GeomDistance(self.geomloss_setting)
        self.pos_is_in_fea =self.opt["pos_is_in_fea",True,"the position info is in the feature this should be consistent with 'include_pos_in_final_feature' in the feature extractor model"]

        self.soften_gt_sigma  =  self.opt[("soften_gt_sigma",0.005,"sigma to soften the one-hot gt")]
        self.buffer = {"gt_one_hot":None, "gt_plan":None,"soften_gt_one_hot":None}



    def naive_corres(self, cur_source, target):
        points = cur_source.points
        pointfea1 = cur_source.pointfea
        pointfea2 = target.pointfea
        weights = cur_source.weights
        sigma = 0.005

        x_i = LazyTensor(points[:, :, None] / sigma)  # BxNx1xD
        x_j = LazyTensor(points[:, None] / sigma)  # Bx1xNxD
        point_dist2 = -x_i.sqdist(x_j)  # BxNxNx1
        point_loggauss = point_dist2 - LazyTensor(point_dist2.logsumexp(dim=2)[..., None])  # BxNxN

        fea_i = LazyTensor(pointfea1[:, :, None])  # BxNx1xd
        fea_j = LazyTensor(pointfea2[:, None])  # Bx1xNxd
        fea_dist2 = -fea_i.sqdist(fea_j)  # BxNxNx1
        fea_logsoftmax = fea_dist2 - LazyTensor(fea_dist2.logsumexp(dim=2)[..., None])  # BxNxNx1

        #l2_loss = (((point_loggauss.exp() - fea_logsoftmax.exp())) ** 2 * weight_i).sum(2)
        ce_loss = -(point_loggauss.exp()*fea_logsoftmax).sum(2)*weights
        return ce_loss.sum(1)[...,0]  # B


    def compute_gt_one_hot(self, refer_points):
        if self.buffer["gt_one_hot"] is not None and self.buffer["gt_one_hot"].shape[0] == refer_points.shape[0]:
            return self.buffer["gt_one_hot"]
        else:
            self.buffer["gt_one_hot"] = None
        if self.buffer["gt_one_hot"] is None:
            B, N = refer_points.shape[0], refer_points.shape[1]
            device = refer_points.device
            i = torch.arange(N,dtype=torch.float32, device=device)
            i = i.repeat(B,1).contiguous()
            j = i
            i = LazyTensor(i[:, :, None, None])
            j = LazyTensor(j[:,None, :, None])
            id_matrix = (.5 - (i - j)**2).step() # BxNxN
            self.buffer["gt_one_hot"] = id_matrix
        return self.buffer["gt_one_hot"]

    def compute_soften_gt_one_hot(self, refer_points):
        sigma = 0.005 * 1.4142135623730951
        x_i = LazyTensor(refer_points[:, :, None] / sigma)  # BxNx1xD
        x_j = LazyTensor(refer_points[:, None] / sigma)  # Bx1xNxD
        point_dist2 = -x_i.sqdist(x_j)  # BxNxNx1
        point_loggauss = point_dist2 - LazyTensor(point_dist2.logsumexp(dim=2)[..., None])  # BxNxN
        return point_loggauss.exp()


    def compute_gt_plan(self,refer_points, refer_weights):
        if self.buffer["gt_plan"] is not None and self.buffer["gt_plan"].shape[0] == refer_points.shape[0]:
            return self.buffer["gt_plan"]
        else:
            self.buffer["gt_plan"] = None

        if self.buffer["gt_plan"] is None:
            gt_one_hot = self.compute_gt_one_hot(refer_points) # BxNxN
            refer_weights = LazyTensor(refer_weights[:,None]) # Bx1xN
            gt_plan = gt_one_hot*refer_weights
            self.buffer["gt_plan"] = gt_plan
        return self.buffer["gt_plan"]




    def ot_corres(self, cur_source, target):
        points = cur_source.points
        weights = cur_source.weights  # BxNx1
        gt_corres = self.compute_gt_plan(points, weights)
        self.geomloss_setting["mode"] = "trans_plan"
        lazy_transport_plan, _ = wasserstein_forward_mapping(cur_source, target, self.geomloss_setting)  # BxNxM
        factor = 1000
        l2_loss = ((factor*(gt_corres - lazy_transport_plan)) ** 2).sum(2)  # BxN
        l2_loss = torch.dot(l2_loss.view(-1), torch.ones_like(l2_loss).view(-1))
        B = points.shape[0]
        return torch.stack([l2_loss / B] * B, 0)


    def ot_ce(self,cur_source, target):
        points = cur_source.points
        weights = cur_source.weights # BxNx1
        lazy_gt_one_hot = self.compute_gt_one_hot(points) if self.loss_type=="ot_ce" else self.compute_soften_gt_one_hot(points)
        self.geomloss_setting["mode"] = "prob"
        _,lazy_log_prob = wasserstein_forward_mapping(cur_source, target, self.geomloss_setting)  #BxNxM
        ce_loss =  -(lazy_gt_one_hot* lazy_log_prob).sum(2)*weights # BxN
        ce_loss = torch.dot(ce_loss.view(-1), torch.ones_like(ce_loss).view(-1))
        B =  points.shape[0]
        factor =0.001
        return torch.stack([ce_loss * factor/B]*B,0)


    def mapped_bary_center(self,cur_source, target):
        self.geomloss_setting["mode"] = "soft"
        flowed = wasserstein_forward_mapping(cur_source, target, self.geomloss_setting)  # BxNxM
        return (((target.points-flowed.points)**2).sum(2)*flowed.weights[...,0]).sum(1)




    def reg_loss(self,cur_source, target):
        if self.pos_is_in_fea:
            reg_s = (cur_source.pointfea[:,:,3:])**2
            reg_t = (target.pointfea[:,:,3:])**2
        else:
            reg_s = (cur_source.pointfea) ** 2
            reg_t = (target.pointfea) ** 2
        return (reg_s+reg_t).mean(2).mean(1)


    def ot_distance(self, cur_source, target):
        return self.geom_loss(cur_source, target)


    def __call__(self,cur_source, target):

        reg_loss = self.reg_loss(cur_source,target)
        if self.loss_type=="ot_map":
            return self.mapped_bary_center(cur_source,target), reg_loss
        if self.loss_type == "ot_distance":
            return self.ot_distance(cur_source, target), reg_loss
        elif self.loss_type == "ot_corres":
            return self.ot_corres(cur_source, target), reg_loss
        elif self.loss_type=="ot_ce" or self.loss_type=="ot_soften_ce":
            return self.ot_ce(cur_source, target), reg_loss
        else:
            return self.naive_corres(cur_source, target), reg_loss





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
