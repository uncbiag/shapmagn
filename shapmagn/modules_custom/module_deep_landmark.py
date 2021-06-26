from copy import deepcopy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from shapmagn.global_variable import Shape
from shapmagn.utils.obj_factory import obj_factory

from shapmagn.modules_reg.networks.pointpwc_multiresol_net import PointConvFeature





class PointConvLandmarkPredictor(nn.Module):
    def __init__(self,opt):
        super(PointConvLandmarkPredictor,self).__init__()
        self.opt = opt
        local_feature_extractor_obj = self.opt[
            ("local_feature_extractor_obj", "", "function object for local_feature_extractor")]
        self.local_feature_extractor = obj_factory(local_feature_extractor_obj) if len(local_feature_extractor_obj) else self.default_local_feature_extractor
        self.input_channel = self.opt[("input_channel",1,"input channel")]
        self.num_landmarks = self.opt[("num_landmarks",6, "num of landmarks")]
        self.predict_at_low_resl = self.opt[("predict_at_low_resl",False, "predict at low resoltuion")]
        self.initial_radius = self.opt[("initial_radius",0.001,"initial radius, only set when use aniso kernel")]
        self.param_shrink_factor = self.opt[("param_shrink_factor",2,"network parameter shrink factor")]
        self.use_aniso_kernel = self.opt[("use_aniso_kernel",False,"use the aniso kernel in first sampling layer")]
        self.normalized_strategy = self.opt["normalized_strategy","sigmoid","sigmoid or lineaer"]
        self.heatmap_threshold = self.opt["heatmap_threshold",0.5,"set heat map value to 0 if < threshold"]
        self.init_deep_landmark_predictor()
        self.buffer = {}
        self.iter = 0

    def default_local_feature_extractor(self, input_shape):
        input_shape.pointfea = input_shape.points.clone()  # torch.cat([cur_source.points, cur_source.weights], 2)
        return input_shape

    def init_deep_landmark_predictor(self):
        self.predictor = PointConvFeature(input_channel=self.input_channel,output_channels=self.num_landmarks,predict_at_low_resl=self.predict_at_low_resl,param_shrink_factor=self.param_shrink_factor, use_aniso_kernel=self.use_aniso_kernel)


    def normalize_heatmap(self, heatmap):
        """
        normalized the heat map into [0,1]
        :param heatmap: BxLxN
        :return:  BxLxN
        """

        if self.normalized_strategy == "sigmoid":
            heatmap = F.sigmoid(heatmap)
            heatmap = heatmap
        elif self.normalized_strategy == "linear":
            B, L = heatmap.shape[:2]
            hm_min = heatmap.min(2)[0].view(B,L,1)
            hm_max = heatmap.max(2)[0].view(B,L,1)
            heatmap = (heatmap-hm_min)/(hm_max-hm_min)
        heatmap[heatmap<self.heatmap_threshold] = 0
        heatmap = heatmap/(heatmap.sum(-1, keepdim=True)+1e-9)
        return heatmap

    def deep_landmark_predictor(self, input_shape):
        heatmap, sampled_points = self.predictor(input_shape.points, input_shape.pointfea) # BxLxN,  BxNxD
        heatmap = self.normalize_heatmap(heatmap) # BxCxN
        landmarks = (heatmap[...,None]*sampled_points[:,None]).sum(2)  # BxLxNx1 Bx1xNxD -> BxLxD
        return landmarks

    def __call__(self,input_shape,batch_info=None):
        input_shape = self.local_feature_extractor(input_shape)
        pred_landmarks = self.deep_landmark_predictor(input_shape)
        output_shape = Shape().set_data_with_refer_to(input_shape.points, input_shape)
        output_shape.landmarks = pred_landmarks
        return output_shape






class DeepLandmarkPredictorLoss(nn.Module):
    def __init__(self, opt):
        super(DeepLandmarkPredictorLoss,self).__init__()
        self.opt = opt
        self.loss_type = self.opt[("loss_type", "mse")]



    def mse(self,pred, gt):
        """

        :param pred: BxLxD
        :param gt: BxLxD
        :return:
        """
        return ((gt-pred)**2).sum(2).mean(1)


    def __call__(self,output_shape, refer_shape,has_gt):
        pred = output_shape.landmarks
        gt = refer_shape.landmarks
        if self.loss_type=="mse":
            sim_loss = self.mse(pred,gt)
            return sim_loss, sim_loss*0
        else:
            raise NotImplemented




class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def grad_hook(grad):
    # import pydevd
    # pydevd.settrace(suspend=False, trace_only_current_thread=True)
    print("debugging info, the grad_norm is {} ".format(grad.norm()))
    return grad



