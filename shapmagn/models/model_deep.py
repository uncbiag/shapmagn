import os
from shapmagn.models.model_base import ModelBase
from shapmagn.global_variable import *
from shapmagn.utils.net_utils import print_model
from shapmagn.utils.shape_visual_utils import save_shape_pair_into_files
from shapmagn.shape.shape_pair_utils import create_shape_pair
from shapmagn.modules.optimizer import optimizer_builder
from shapmagn.modules.scheduler import scheduler_builder
class DeepModel(ModelBase):
    """
    Deep models
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


        self.opt_optim = opt[('optim', {}, "setting for the optimizer")]
        self.opt_scheduler = opt[('scheduler', {}, "setting for the scheduler")]
        self.criticUpdates = opt['tsk_set']['criticUpdates']
        self.init_learning_env(self.opt,device)
        self.step_count = 0
        """ count of the step"""
        self.cur_epoch = 0
        prepare_input_object = opt[("prepare_input_object", "", "input processing function")]
        self.prepare_input = obj_factory(prepare_input_object) if prepare_input_object else self._set_input
        source_target_generator = opt[
            ("source_target_generator", "shape_pair_utils.create_source_and_target_shape()", "generator func")]
        self.source_target_generator = obj_factory(source_target_generator)
        capture_plotter_obj = opt[
            ("capture_plot_obj", "visualizer.capture_plotter()", "factory object for 2d capture plot")]
        self.capture_plotter = obj_factory(capture_plotter_obj)
        analyzer_obj = opt[
            ("analyzer_obj", "utils.compute_jacobi_of_pointcloud()", "result analyzer")]
        self.analyzer = obj_factory(analyzer_obj)


    def init_learning_env(self, opt, device):
        method_name = opt[('method_name', "discrete_flow_deep", "specific optimization method")]
        if method_name in ["feature_deep","discrete_flow_deep"]:
            method_opt = opt[(method_name, {}, "method settings")]
            self._model = MODEL_POOL[method_name](method_opt).to(device)
        else:
            raise ValueError("method not supported")

        # if gpus and len(gpus) >= 1:
        #     self._model = nn.DataParallel(self._model, gpus)
        self.optimizer = optimizer_builder(self.opt_optim)(self._model.parameters())
        self.lr_scheduler = scheduler_builder(self.opt_scheduler)(self.optimizer)
        self.shape_folder_3d = os.path.join(self.record_path, "3d")
        os.makedirs(self.shape_folder_3d, exist_ok=True)
        self.shape_folder_2d = os.path.join(self.record_path, "2d")
        os.makedirs(self.shape_folder_2d, exist_ok=True)

        if self._model is not None:
            print('---------- A model instance for {} is initialized -------------'.format(method_name))
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


    def _set_input(self, input_data, batch_info):
        return input_data, batch_info

    def set_input(self, input_data, device, is_train=False):
        batch_info = {"pair_name":input_data["pair_name"],
                           "source_info":input_data["source_info"],
                           "target_info":input_data["target_info"]}
        input_data, self.batch_info =  self.prepare_input(input_data, batch_info)
        return input_data




    def get_debug_info(self):
        """ get filename of the failed cases"""
        info = {'file_name': self.batch_info["fname_list"]}
        return info

    def forward(self, shape_pair=None):
        self._model.module.set_cur_epoch(self.cur_epoch)
        loss, shape_pair = self._model(shape_pair)
        return loss, shape_pair


    def backward_net(self, loss):
        loss.backward()

    def optimize_parameters(self, input_data=None):
        """
        forward and backward the model, optimize parameters and manage the learning rate
        :return:
        """
        source, target = self.source_target_generator(input_data)
        shape_pair = create_shape_pair(source, target)
        if self.is_train:
            self.iter_count += 1
        output = self.forward(shape_pair)
        loss = output[0].mean()
        self.backward_net(loss / self.criticUpdates)
        self.loss = loss.item()
        update_lr, lr = self._model.module.check_if_update_lr()
        if update_lr:
            self.update_learning_rate(lr)
        if self.iter_count % self.criticUpdates == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return shape_pair

    def update_scheduler(self,epoch):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(epoch)

        for param_group in self.optimizer.param_groups:
            print("the current epoch is {} with learining rate set at {}".format(epoch,param_group['lr']))


    def get_evaluation(self,input_data):
        loss, shape_pair = self.optimize_parameters(input_data)
        return shape_pair



    def save_visual_res(self, input_data, shape_pair, phase):
        save_shape_pair_into_files(self.shape_folder_3d, "{}_epoch_{}".format(phase,self.cur_epoch), shape_pair)
        self.capture_plotter(self.shape_folder_2d, "{}_epoch_{}".format(phase, self.cur_epoch), shape_pair)





    def analyze_res(self, res):
        self.analyzer(res, self.batch_info["pair_name"], self._model, self.record_path,)




    def save_res(self,phase, saving=True):
        pass


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
