import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from shapmagn.models.model_base import ModelBase
from shapmagn.global_variable import *
from shapmagn.utils.obj_factory import obj_factory
from shapmagn.utils.net_utils import print_model
from shapmagn.metrics.losses import Loss
from shapmagn.models.multiscale_optimization import build_multi_scale_solver

class OptModel(ModelBase):
    """ Optimization models """

    def name(self):
        return 'Optimization Model'

    def initialize(self, opt,device, gpus=None):
        """
        initialize variable settings of Optimization Approches
        multi-gpu is not supported for optimization tasks

        :param opt: ParameterDict, task settings
        :return:
        """
        ModelBase.initialize(self,opt, device, gpus)
        method_name= opt['method_name']
        self._model = MODEL_POOL[method_name](opt)
        """create a model with given method"""
        self.criticUpdates = opt['criticUpdates']
        loss_opt = opt[("loss",{},"settings for loss")]
        loss_fn = Loss(loss_opt)
        self._model.set_loss_fn(loss_fn)
        if gpus and len(gpus) >= 1:
            self._model = nn.DataParallel(self._model, gpus)
        self._model.to(device)
        self.step_count = 0
        """ count of the step"""
        self.cur_epoch = 0
        """visualize condition"""
        self.visualize_condition = {}
        print('---------- A model instance on {} is initialized -------------'.format(method_name))
        print_model(self._model)
        print('-----------------------------------------------')




    def set_input(self, input_data, device, is_train=True):
        """
        :param input_data:
        :param is_train:
        :return:
        """
        self.batch_info = {"pair_name":input_data["pair_name"],
                           "source_info":input_data["source_info"],
                           "target_info":input_data["target_info"]}

        reg_param_initializer = obj_factory(self.opt["init_reg_param","default_shape_pair.init_reg_param(input_data)",
                                                     "functions that take take 'data_input' as the input and return the registration parameters"])
        input_data["source"] = input_data["source"].to(device)
        input_data["target"] = input_data["target"].to(device)
        reg_param = reg_param_initializer(input_data)
        input_data["reg_param"] = reg_param.to(device)
        return input_data



    def backward_net(self, loss):
        loss.backward()

    def get_debug_info(self):
        """ get filename of the failed cases"""
        info = {'file_name': self.batch_info["fname_list"]}
        return info


    # def optimize_parameters(self, input_data=None):
    #     """
    #     forward and backward the model, optimize parameters and manage the learning rate
    #
    #     :param input_data: input_data(not used
    #     :return:
    #     """
    #     self.opt_optim = self.opt['optim']
    #     """settings for the optimizer"""
    #     self.optimizer = optimizer_builder(self.opt_optim)([input_data["reg_param"]])
    #     self.lr_scheduler = scheduler_builder(self.optimizer)
    #     """initialize the optimizer and scheduler"""
    #
    #     output = self.forward(input_data)
    #     loss = output[0].mean()
    #     self.backward_net(loss / self.criticUpdates)
    #     self.loss = loss.item()
    #     update_lr, lr = self._model.module.check_if_update_lr()
    #     if update_lr:
    #         self.update_learning_rate(lr)
    #     if self.iter_count % self.criticUpdates == 0:
    #         self.optimizer.step()
    #         self.optimizer.zero_grad()

    def optimize_parameters(self, input_data=None):
        """
        forward and backward the model, optimize parameters and manage the learning rate

        :param input_data: input_data(not used
        :return:
        """
        sovler = build_multi_scale_solver(self.opt,self._model)
        output = sovler(input_data)



    def get_evaluation(self,input_data):
        def get_vid_prob_seq(vid,labels):
            """
            get the prob sequence for each frame in video
            :param vid: a tensor NxCxHxW
            :return prob: a tensor Nx1
            """
            frames_split = torch.split(vid, eval_phase_bch_sz)
            labels_split = torch.split(labels, eval_phase_bch_sz) if labels is not None else [None]*len(frames_split)
            prob = []
            for sub_frames,sub_labels in zip(frames_split,labels_split):
                sub_output = self.forward((sub_frames,sub_labels))
                sub_prob = sub_output[1]
                prob.append(sub_prob.detach().cpu())
            prob = torch.cat(prob,dim=0)
            return prob

        print("Processing video: {}".format(self.batch_info["fname_list"]))
        vid_list, label_list = input_data
        eval_phase_bch_sz = self.opt[('eval_phase_inner_loop_bch_sz',1,"batch size for evaluation phase")]
        with torch.no_grad():
            vid_prob_list = [get_vid_prob_seq(vid, label) for vid, label in zip(vid_list, label_list)]
        metric_dict, prediction_list, prob_list = get_binary_video_metric(vid_prob_list, label_list, thres=[0.1,0.3,0.5,0.7])
        return metric_dict, prediction_list, prob_list



    def save_visual_res(self, save_visual_results, input_data, eval_res, phase):
        if not save_visual_results:
            return
        self.save_shapmagn(input_data,eval_res, phase)




    def check_visual_condition(self,phase, case):
        max_visual_per_case = self.opt["tsk_set"]["visual"][("max_visual_per_case",10,"max num of videos to save per class")]
        return True
        if max_visual_per_case <= 0:
            return False
        name = phase + "_", str(self.cur_epoch)
        if name not in self.visualize_condition:
            self.visualize_condition[name]={}
        if case not in self.visualize_condition[name]:
            self.visualize_condition[name][case] = 1
            return True
        else:
            if self.visualize_condition[name][case] < max_visual_per_case:
                self.visualize_condition[name][case] += 1
                return True
        return False




    def analyze_res(self, res, cache_res=True):
        metric_dict, prediction_list, prob_list = res
        if cache_res:
            if "pred" not in self.caches:
                self.caches.update({"pred":prediction_list, "fname":self.batch_info["fname_list"],"prob":prob_list})
            else:
                self.caches['pred'] += prediction_list
                self.caches['fname'] += self.batch_info["fname_list"]
                self.caches['prob'] += prob_list
        if len(metric_dict):
            return metric_dict["loss"], metric_dict
        else:
            return -1, np.array([-1])



    def save_res(self,phase, saving=True):
        if saving:
            saving_pred_path = os.path.join(self.record_path,"predictions_{}_{}.csv".format(phase,self.cur_epoch))
            submission_df = pd.DataFrame({"filename": self.caches['fname'], "label": self.caches['pred']})
            submission_df.to_csv(saving_pred_path, index=False)
            saving_prob_path = os.path.join(self.record_path, "pro_{}_{}.npy".format(phase, self.cur_epoch))
            np.save(saving_prob_path,np.array(self.caches['prob']))
        self.caches = {}


    def get_extra_to_plot(self):
        """
        extra image to be visualized

        :return: image (BxCxXxYxZ), name
        """
        return self._model.get_extra_to_plot()


    def set_test(self):
        torch.set_grad_enabled(True)


