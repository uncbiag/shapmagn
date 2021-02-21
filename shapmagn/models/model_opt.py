import os
import numpy as np
from shapmagn.models.model_base import ModelBase
from shapmagn.global_variable import *
from shapmagn.utils.net_utils import print_model
from shapmagn.models.multiscale_optimization import build_multi_scale_solver
from shapmagn.utils.shape_visual_utils import save_shape_pair_into_files
from shapmagn.shape.shape_pair_utils import create_shape_pair

class OptModel(ModelBase):
    """
    Optimization models include two step optimization: affine and non-parametric optimization
    the affine optimization (optional, Teaser, a scaled rigid reigstration approach, for default),
    ('true' affine approaches needs to be implemented)
     currently affine optimization is not warped by multi-scale optimization framework)
     # todo implement gradient flow based affine param regression for cases that noise level is low
    the non-parametric optimization (optional, gradient flow for default) is implemented multi-scale optimization framework


    """

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
        self.init_optimization_env(self.opt,device)
        self.step_count = 0
        """ count of the step"""
        self.cur_epoch = 0
        """visualize condition"""
        self.visualize_condition = {}
        source_target_generator = opt[
            ("source_target_generator", "shape_pair_utils.create_source_and_target_shape()", "generator func")]
        self.source_target_generator = obj_factory(source_target_generator)



    def init_optimization_env(self, opt, device):
        method_name = opt[('method_name', "lddmm_opt", "specific optimization method")]
        prealign_opt = opt[("prealign_opt",{}, "method settings")]
        if method_name in ["lddmm_opt","discrete_flow_opt","gradient_flow_opt"]:
            self.run_prealign = False
            self.run_nonparametric = True
            self._prealign_model = None
            method_opt = opt[(method_name, {}, "method settings")]
            self._model = MODEL_POOL[method_name](method_opt).to(device)
        elif method_name == "prealign_opt":
            self.run_prealign = True
            self.run_nonparametric = False
            self._prealign_model = MODEL_POOL["prealign_opt"](prealign_opt).to(device)
            self._model = None
        elif "_and_prealign_opt" in method_name:
            self.run_prealign = True
            self.run_nonparametric = True
            self._prealign_model = MODEL_POOL["prealign_opt"](prealign_opt).to(device)
            method_name = method_name.replace("_and_prealign_opt","")
            method_opt = opt[(method_name, {}, "method settings")]
            self._model = MODEL_POOL[method_name](method_opt).to(device)
        # if gpus and len(gpus) >= 1:
        #     self._model = nn.DataParallel(self._model, gpus)
        if self._prealign_model is not None:
            print('---------- A model instance for {} is initialized -------------'.format("prealign_opt"))
            print_model(self._prealign_model)
            print('-----------------------------------------------')
        if self._model is not None:
            print('---------- A model instance for {} is initialized -------------'.format(method_name))
            print_model(self._model)
            print('-----------------------------------------------')

    def set_input(self, input_data, device, is_train=False):
        """
        :param input_data:
        :param is_train:
        :return:
        """
        self.batch_info = {"pair_name":input_data["pair_name"],
                           "source_info":input_data["source_info"],
                           "target_info":input_data["target_info"]}
        input_data["source"] = {key: fea.to(device) for key, fea in input_data["source"].items()}
        input_data["target"] = {key: fea.to(device) for key, fea in input_data["target"].items()}
        return input_data


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

    def optimize_parameters(self, data=None):
        """
        forward and backward the model, optimize parameters and manage the learning rate

        :param data: input_data(not used
        :return:
        """
        source, target = self.source_target_generator(data)
        shape_pair = create_shape_pair(source, target)
        if self.run_prealign:
            multi_scale_opt = self.opt[("multi_scale_optimization_prealign",{},"settings for multi_scale_optimization_prealign")]
            multi_scale_opt['record_path'] = self.record_path
            sovler = build_multi_scale_solver(multi_scale_opt,self._prealign_model)
            shape_pair = sovler(shape_pair)
            save_shape_pair_into_files(self.record_path, "shape_affined", shape_pair)
            if self.run_nonparametric:
                shape_pair.control_points = shape_pair.flowed_control_points
        if self.run_nonparametric:
            multi_scale_opt = self.opt[("multi_scale_optimization",{},"settings for multi_scale_optimization")]
            multi_scale_opt['record_path'] = self.record_path
            sovler = build_multi_scale_solver(multi_scale_opt,self._model)
            shape_pair = sovler(shape_pair)
            save_shape_pair_into_files(self.record_path, "shape_nonparametric", shape_pair)
        return shape_pair



    def get_evaluation(self,input_data):
        output = self.optimize_parameters(input_data)
        return None, None



    def save_visual_res(self, save_visual_results, input_data, eval_res, phase):
        if not save_visual_results:
            return
        pass




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


