import torch.nn as nn
from shapmagn.utils.obj_factory import obj_factory
from shapmagn.modules.networks.pointnet2.util import PointNetSetAbstraction, PointNetSetUpConv, PointNetFeaturePropogation
from shapmagn.metrics.losses import GeomDistance
from shapmagn.modules.gradient_flow_module import wasserstein_forward_mapping



class PointNet2FeaExtractor(nn.Module):
    def __init__(self,opt):
        super(PointNet2FeaExtractor,self).__init__()
        self.opt = opt
        self.local_pair_feature_extractor = obj_factory(self.opt[("local_pair_feature_extractor_obj","","function object for local_feature_extractor")])
        self.init_deep_feature_extractor()
        self.include_pos =  self.opt[("include_pos",True,"include_pos")]
        self.buffer = {}
        self.iter = 0


    def init_deep_feature_extractor(self):
        self.sa1 = PointNetSetAbstraction(npoint=4096, radius=0.01, nsample=16, in_channel=3, mlp=[32,32,64], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=2048, radius=0.02, nsample=16, in_channel=64, mlp=[64, 64, 128], group_all=False)
        self.su2 = PointNetSetUpConv(nsample=8, radius=0.6, f1_channel = 64, f2_channel = 256, mlp=[128, 128, 256], mlp2=[256])
        self.fp = PointNetFeaturePropogation(in_channel = 256+3, mlp = [256, 256])


    def deep_pair_feature_extractor(self,cur_source, target):
        pc1, pc2, feature1, feature2 = cur_source.points, target.points, cur_source.pointfea, target.pointfea

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


        cur_source_deepfea = self.deep_feature_extractor(cur_source_points, cur_source_pointfea)
        target_deepfea = self.deep_feature_extractor(target_points, target_pointfea)

    def normalize_fea(self):
        pass

    def __call__(self,cur_source, target, iter=-1):
        cur_source, target = self.local_pair_feature_extractor(cur_source, target)
        cur_source.pointfea, target.pointfea = cur_source_deepfea, target_deepfea
        return cur_source, target




class DeepFeatureLoss(nn.Module):
    def __init__(self, opt):
        super(DeepFeatureLoss,self).__init__()
        self.opt = opt
        self.loss_type = self.opt[("loss_type", "ot_corres","naive_corres/ ot_corres/ ot_distace")]
        self.geom_loss = GeomDistance(self.opt["geomloss_setting"])


    def naive_corres(self, cur_source, target):
        points = cur_source.points
        pointfea1 = cur_source.pointfea
        pointfea2 = target.pointfea
        weight_i = LazyTensor(weights[:, :, None])  # BxNx1x1
        sigma = 0.05

        x_i = LazyTensor(points[:, :, None] / sigma)  # BxNx1xD
        x_j = LazyTensor(points[:, None] / sigma)  # Bx1xNxD
        point_dist2 = -x_i.sqdist(x_j)  # BxNxNx1
        point_loggauss = point_dist2 - LazyTensor(point_dist2.logsumexp(dim=2)[..., None])  # BxNxN

        fea_i = LazyTensor(pointfea1[:, :, None])  # BxNx1xd
        fea_j = LazyTensor(pointfea2[:, None])  # Bx1xNxd
        fea_dist2 = -fea_i.sqdist(fea_j)  # BxNxNx1
        fea_logsoftmax = fea_dist2 - LazyTensor(fea_dist2.logsumexp(dim=2)[..., None])  # BxNxNx1

        l2_loss = (((point_loggauss.exp() - fea_logsoftmax.exp())) ** 2 * weight_i).sum(2)
        # ce_loss = -(point_loggauss.exp()*fea_logsoftmax).sum(2)*weight_i
        return l2_loss.sum(1)  # B


    def ot_corres(self,cur_source, target):
        points = cur_source.points
        weights = cur_source.weights # BxNx1
        batch, npoints = points.shape[0], points.shape[1]
        gt_corres = torch.diag_embed(torch.ones(batch,npoints,1, device=points.device)*weights)
        lazy_transport_plan = wasserstein_forward_mapping(cur_source, target, self.geomloss_setting)  #BxNx1
        l2_loss =  (((gt_corres - lazy_transport_plan)) ** 2 * weights).sum(2).sum(1)
        return l2_loss


    def ot_distance(self, cur_source, target):
        return self.geom_loss(cur_source, target)


    def __call__(self,cur_source, target):
        if self.loss_type == "ot_distance":
            return self.ot_distance(cur_source, target)
        elif self.loss_type == "ot_corres":
            return self.ot_corres(cur_source, target)
        else:
            return self.naive_corres(cur_source, target)





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
