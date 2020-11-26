import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from shapmagn.models.model_base import ModelBase
from shapmagn.global_variable import *
from shapmagn.utils.net_utils import print_model
from shapmagn.metrics.losses import Loss
from shapmagn.modules.optimizer import build_optimizer
from shapmagn.modules.scheduler import build_scheduler


class OptModel(ModelBase):
    """ Optimization models """

    def name(self):
        return 'Optimization Model'

    def initialize(self, opt,device, gpus):
        """
        initialize variable settings of Optimization Approches

        :param opt: ParameterDict, task settings
        :return:
        """
        ModelBase.initialize(self,opt, device, gpus)
        method_name= opt['method_name']
        self._model = model_pool[method_name](opt)
        """create a model with given method"""
        self.criticUpdates = opt['criticUpdates']
        loss_name = opt['loss_name']
        loss_opt = opt[("loss",{},"settings for loss")]
        loss_fn = Loss(loss_opt)
        self._model.set_loss_fn(loss_fn)
        if gpus and len(gpus) >= 1:
            self._model = nn.DataParallel(self._model, gpus)
        self._model.to(device)
        self.opt_optim = opt['optim']
        """settings for the optimizer"""
        self.init_optimize_instance(warmming_up=True)
        """initialize the optimizer and scheduler"""
        self.step_count = 0
        """ count of the step"""
        self.cur_epoch = 0
        """visualize condition"""
        self.visualize_condition = {}
        print('---------- A model instance on {} is initialized -------------'.format(method_name))
        print_model(self._model)
        print('-----------------------------------------------')


    def update_learning_rate(self, new_lr=-1):
        """
        set new learning rate

        :param new_lr: new learning rate
        :return:
        """
        if new_lr < 0:
            lr = self.opt_optim['lr']
        else:
            lr = new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        print(" the learning rate now is set to {}".format(lr))

    def set_input(self, input_data, device, is_train=True):
        """
        :param input_data:
        :param is_train:
        :return:
        """
        self.batch_info = {"pair_name":input_data["pair_name"],
                           "source_info":input_data["source_info"],
                           "target_info":input_data["target_info"]}
        source, target = input_data["source"], input_data["target"]
        return source, target

    def init_optimize_instance(self, warmming_up=False):
        """ get optimizer and scheduler instance"""
        self.optimizer, self.lr_scheduler, self.exp_lr_scheduler = self.init_optim(self.opt_optim, self._model,
                                                                                   warmming_up=warmming_up)


    def init_optim(self, opt, _model, warmming_up=False):
        """
        set optimizers and scheduler

        :param opt: settings on optimizer
        :param _model: model with learnable parameters
        :param warmming_up: if set as warmming up
        :return: optimizer, custom scheduler, plateau scheduler
        """
        optimize_name = opt['optim_type']
        lr = opt['lr']
        beta = opt['adam']['beta']
        lr_sched_opt = opt[('lr_scheduler',{},"settings for learning scheduler")]
        self.lr_sched_type = lr_sched_opt['type']
        if optimize_name == 'adam':
            re_optimizer = torch.optim.Adam(_model.parameters(), lr=lr, betas=(beta, 0.999))
        else:
            re_optimizer = torch.optim.SGD(_model.parameters(), lr=lr)
        re_optimizer.zero_grad()
        re_lr_scheduler = None
        re_exp_lr_scheduler = None
        if self.lr_sched_type == 'custom':
            step_size = lr_sched_opt['custom'][('step_size',50,"update the learning rate every # epoch")]
            gamma = lr_sched_opt['custom'][('gamma',0.5,"the factor for updateing the learning rate")]
            re_lr_scheduler = torch.optim.lr_scheduler.StepLR(re_optimizer, step_size=step_size, gamma=gamma)
        elif self.lr_sched_type == 'plateau':
            patience = lr_sched_opt['plateau']['patience']
            factor = lr_sched_opt['plateau']['factor']
            threshold = lr_sched_opt['plateau']['threshold']
            min_lr = lr_sched_opt['plateau']['min_lr']
            re_exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(re_optimizer, mode='min', patience=patience,
                                                                 factor=factor, verbose=True,
                                                                 threshold=threshold, min_lr=min_lr)
        if not warmming_up:
            print(" no warming up the learning rate is {}".format(lr))
        else:
            lr = opt['lr']/10
            for param_group in re_optimizer.param_groups:
                param_group['lr'] = lr
            re_lr_scheduler.base_lrs = [lr]
            print(" warming up on the learning rate is {}".format(lr))
        return re_optimizer, re_lr_scheduler, re_exp_lr_scheduler

    def backward_net(self, loss):
        loss.backward()

    def get_debug_info(self):
        """ get filename of the failed cases"""
        info = {'file_name': self.batch_info["fname_list"]}
        return info

    def forward(self, input_data=None):
        """

        :param input_data(not used )
        :return: warped image intensity with [-1,1], transformation map defined in [-1,1], affine image if nonparameteric reg else affine parameter
        """
        self._model.module.set_cur_epoch(self.cur_epoch)
        output = self._model(input_data, self.is_train)
        return output

    def update_scheduler(self,epoch):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(epoch)

        for param_group in self.optimizer.param_groups:
            print("the current epoch is {} with learining rate set at {}".format(epoch,param_group['lr']))

    def optimize_parameters(self, input_data=None):
        """
        forward and backward the model, optimize parameters and manage the learning rate

        :param input_data: input_data(not used
        :return:
        """

        if self.is_train:
            self.iter_count += 1
        output = self.forward(input_data)
        loss = output[0].mean()
        self.backward_net(loss / self.criticUpdates)
        self.loss = loss.item()
        update_lr, lr = self._model.module.check_if_update_lr()
        if update_lr:
            self.update_learning_rate(lr)
        if self.iter_count % self.criticUpdates == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()




    def get_current_errors(self):
        return self.loss




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


    def save_shapmagn(self,input_data, eval_res, phase):
        from shapmagn.utils.net_utils import VisualWarpper
        metric_dict, prediction_list, prob_list = eval_res
        inverse_prob_list = [(1-prob).squeeze() for prob in prob_list]  # here we interested in class 0
        vid_list, labels_list = input_data
        label_list = [torch.unique(vid_label).item() for vid_label in labels_list]
        is_acc_list = [label==pred for label, pred in zip(label_list, prediction_list)]
        f_info_list = [{'fname': fn, "fpath": fpath, "seg_path":seg_path, "lms_path": lms_path, "frame_index": fid}
                       for fn, fpath, seg_path, lms_path, fid in
                       zip(self.batch_info["fname_list"], self.batch_info["fpath_list"], self.batch_info["seg_path_list"],
                           self.batch_info["lms_path_list"], self.batch_info["frame_index_list"])]
        num_vid = len(vid_list)
        visual_mode = self.opt["visual"][('visual_mode',"integrated", "mode of visualization: integrated/gradcam")]

        def get_vid_act_map(vid, labels, ig, f_info=None):
            """
            get the prob sequence for each frame in video
            :param vid: a tensor NxCxHxW
            :return prob: a tensor Nx1
            """
            frames_split = torch.split(vid, train_phase_bch_sz)
            labels_split = torch.split(labels, train_phase_bch_sz) if labels is not None else [None] * len(frames_split)
            attribution_list = []
            center_frame_id = vid.shape[1]//2
            if visual_mode == "integrated":
                for sub_frames, sub_labels in zip(frames_split, labels_split):
                    baseline = torch.zeros_like(sub_frames)
                    sub_attri = ig.attribute(sub_frames, baseline, return_convergence_delta=False, n_steps=10,internal_batch_size=train_phase_bch_sz)
                    sub_attri_np = sub_attri.detach().cpu().numpy()
                    sub_attri_np = np.sum(sub_attri_np[:, center_frame_id], axis=1)
                    attribution_list.append(sub_attri_np)

            elif visual_mode == "gradcam":
                for sub_frames, sub_labels in zip(frames_split, labels_split):
                    sub_attri = ig.attribute(sub_frames,attribute_to_layer_input=True)
                    sub_attri = LayerAttribution.interpolate(sub_attri[0], (sub_frames.shape[-2], sub_frames.shape[-1]))
                    sub_attri_np = sub_attri.detach().cpu().numpy()
                    sub_attri_np = sub_attri_np[:,0]
                    attribution_list.append(sub_attri_np)

            elif visual_mode == "fea_ablation":
                from shapmagn.utils.seg_utils import decode_binary_mask
                from shapmagn.utils.landmarks_utils import get_structure_map
                frame_index, lms_path, seg_path = f_info["frame_index"], f_info["lms_path"], f_info["seg_path"]
                assert len(lms_path)>0 and len(seg_path)>0
                seq_lms = np.load(lms_path)['landmarks'][frame_index]
                assert os.path.isfile(seg_path)
                seq_seg = np.load(seg_path, allow_pickle=True)
                seq_seg = [decode_binary_mask(seq_seg[i]) for i in frame_index]
                structure_map = [get_structure_map(seg, lms)[None][None] for seg, lms in zip(seq_seg, seq_lms)]
                structure_map_split = torch.split(torch.Tensor(structure_map).long().to(self.devices), train_phase_bch_sz)
                for sub_frames, sub_structure_map_split in zip(frames_split, structure_map_split):
                    sub_attri = ig.attribute(sub_frames, feature_mask=sub_structure_map_split)
                    sub_attri_np = sub_attri.detach().cpu().numpy()
                    sub_attri_np = sub_attri_np[:, 0]
                    attribution_list.append(sub_attri_np)
            attributions = np.concatenate(attribution_list,0)
            return attributions

        print("Computing activation map of  video: {}".format(self.batch_info["fname_list"]))
        visual_net = VisualWarpper(self._model)
        ig = None
        if visual_mode == "integrated":
            ig = IntegratedGradients(visual_net)
        elif visual_mode == "gradcam":
            ig = LayerGradCam(visual_net,visual_net._model.module.avg_pool, device_ids=self.gpu_ids)
        elif visual_mode == "fea_ablation":
            ig = FeatureAblation(visual_net)
        train_phase_bch_sz = self.opt['batch_sz'][0]

        for i in range(num_vid):
            suffix = "_true" if is_acc_list[i] else "_false"
            case = classes[prediction_list[i]]+suffix
            is_saving = self.check_visual_condition(phase, case)
            if is_saving:
                vid_act_maps = get_vid_act_map(vid_list[i], labels_list[i], ig, f_info_list[i])
                saving_folder = os.path.join(self.record_path, str(self.cur_epoch),visual_mode, case)
                saving_func = save_attri_map_into_video if visual_mode=="fea_ablation" else save_video_with_activation_map
                saving_func(vid_act_maps, inverse_prob_list[i], saving_folder, f_info_list[i])



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

    def set_train(self):
        self._model.train(True)
        self.is_train = True
        torch.set_grad_enabled(True)

    def set_val(self):
        self._model.train(False)
        self.is_train = False
        torch.set_grad_enabled(False)

    def set_debug(self):
        self._model.train(False)
        self.is_train = False
        torch.set_grad_enabled(False)

    def set_test(self):
        self._model.train(False)
        self.is_train = False
        torch.set_grad_enabled(False)
