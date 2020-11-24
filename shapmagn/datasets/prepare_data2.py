
import os
import copy
from shapmagn.datasets.data_utils import generate_pair_name, get_extra_info_path_list, saving_pair_info, divide_sess_set
sesses = ['train', 'val', 'test', 'debug']
number_of_workers = 10
warning_once = True
import random


class BaseRegDataSet(object):

    def __init__(self):
        """
        :param name: name of data set
        :param data_path: path of the dataset
        """

        self.data_path = None
        """path of the dataset"""
        self.output_path = None
        """path of the output directory"""
        self.max_train_pairs = -1
        self.max_total_pairs = -1
        self.divided_ratio = (0.7, 0.1, 0.2)
        """divided the data into train, val, test set"""

    def generate_pair_list(self):
        pass

    def set_data_path(self, path):
        self.data_path = path

    def set_output_path(self, path):
        self.output_path = path
        os.makedirs(path,exist_ok=True)

    def set_divided_ratio(self, ratio):
        self.divided_ratio = ratio

    def save_pair_to_txt(self, info=None):
        pass

    def gen_pair_dic(self):
        pass

    def check_settings(self):
        pass

    def prepare_data(self):
        """
        preprocessig  data for each dataset
        :return:
        """
        self.check_settings()
        print("start preparing data..........")
        print("the output file path is: {}".format(self.output_path))
        info_dict = self.gen_pair_dic()
        self.save_pair_to_txt(copy.deepcopy(info_dict))
        print("data preprocessing finished")


class CustomDataSet(BaseRegDataSet):
    """
    """

    def __init__(self):
        BaseRegDataSet.__init__(self)
        self.reg_coupled_pair = False
        # set either coupled_pair_path_list or self_cross_path_list
        self.coupled_pair_path_list = None
        self.self_cross_path_list = None
        self.extra_info_pair_path_list = None
        self.id_sess_dic = None
        self.has_extra_info = False


    def set_coupled_pair_path_list(self, coupled_pair_path_list):
        """
        [source_path_list, target_path_list], where source and the target has one to one correspondence
        :param coupled_pair_path_list:
        :return:
        """
        self.coupled_pair_path_list = coupled_pair_path_list
        self.reg_coupled_pair = True

    def set_self_cross_path_list(self, self_cross_path_list):
        """
        a path list that pairs are randomly selected from all possible pairs
        :param self_cross_path_list:
        :return:
        """
        self.self_cross_path_list = self_cross_path_list
        self.reg_coupled_pair = False

    def set_extra_info_pair_path_list(self,extra_info_pair_path_list):
        """
        [extra_info_source_path_list, extra_info_target_path_list], set "None" for non-exist item,
        :param extra_info_pair_path_list:
        :return:
        """
        self.extra_info_pair_path_list = extra_info_pair_path_list
        self.has_extra_info = True

    def set_id_sess_dic(self,id_sess_dic):
        """
        {"train": id_list, "val":id_list, "test":id_list, "debug": id_list}
        :return:
        """
        self.id_sess_dic = id_sess_dic

    def check_settings(self):
        if self.reg_coupled_pair:
            assert self.coupled_pair_path_list is not None and self.self_cross_path_list is None
        if not self.reg_coupled_pair:
            assert self.coupled_pair_path_list is None and self.self_cross_path_list is not None



    def __gen_path_and_name_dic(self, pair_list_dic):
        def __gen_pair_name_list(pair_list_dic):
            return {sess: [generate_pair_name([path[0], path[1]]) for path in pair_list_dic[sess]] for sess
                in sesses}
        divided_path_and_name_dic = {}
        divided_path_and_name_dic['pair_path_list'] = pair_list_dic
        divided_path_and_name_dic['pair_name_list'] = __gen_pair_name_list(pair_list_dic)
        return divided_path_and_name_dic




    def __gen_pair_list_from_two_list(self, pair_path_list,extra_info_pair_path_list=None, pair_num_limit=1000):
        obj_path_list_1, obj_path_list_2 = pair_path_list
        if self.has_extra_info:
            extra_info_path_list_1, extra_info_path_list_2 = extra_info_pair_path_list
        pair_path_list = []
        extra_info_pair_path_list = []
        num_obj_1 = len(obj_path_list_1)
        num_obj_2 = len(obj_path_list_2)
        for i in range(num_obj_1):
            count_max = -1  # -1
            pair_path_list_tmp = []
            extra_info_pair_path_list_tmp = []
            for j in range(num_obj_2):
                if generate_pair_name(obj_path_list_1[i]) == generate_pair_name(obj_path_list_2[j]):
                    continue
                pair_path_list_tmp.append([obj_path_list_1[i], obj_path_list_2[j]])
                if self.has_extra_info:
                    extra_info_pair_path_list_tmp.append([ extra_info_path_list_1[i], extra_info_path_list_2[j]])
            if len(pair_path_list_tmp) > count_max and count_max>0:
                rand_ind = random.sample(list(range(len(pair_path_list_tmp))), count_max)
                pair_path_list_tmp = [pair_path_list_tmp[ind] for ind in rand_ind]
                extra_info_pair_path_list_tmp = [extra_info_pair_path_list_tmp[ind] for ind in rand_ind]
            pair_path_list += pair_path_list_tmp
            extra_info_pair_path_list += extra_info_pair_path_list_tmp
        if pair_num_limit >= 0:
            random.shuffle(pair_path_list)
            num_limit = min(len(pair_path_list), pair_num_limit)
            return pair_path_list[:num_limit]
        else:
            return pair_path_list

    def __gen_pair_list_with_coupled_list(self, pair_path_list,extra_info_pair_path_list=None, pair_num_limit=1000):
        obj_path_list_1, obj_path_list_2 = pair_path_list
        if self.has_extra_info:
            extra_info_path_list_1, extra_info_path_list_2 = extra_info_pair_path_list
        obj_pair_list = []
        num_obj_1 = len(obj_path_list_1)
        num_obj_2 = len(obj_path_list_2)
        assert num_obj_1 == num_obj_2
        for i in range(num_obj_1):
            if generate_pair_name(obj_path_list_1[i]) == generate_pair_name(obj_path_list_2[i]):
                continue
            if self.has_extra_info:
                obj_pair_list.append([obj_path_list_1[i], obj_path_list_2[i],
                                      extra_info_path_list_1[i], extra_info_path_list_2[i]])
            else:
                obj_pair_list.append([obj_path_list_1[i], obj_path_list_2[i]])
        if pair_num_limit >= 0:
            random.shuffle(obj_pair_list)
            num_limit = min(obj_pair_list, pair_num_limit)
            return obj_pair_list[:num_limit]
        else:
            return obj_pair_list


    def gen_pair_dic(self):
        if not self.reg_coupled_pair:
            obj_path_list = self.self_cross_path_list
            pair_path_list = [obj_path_list, obj_path_list]
            gen_pair_list_func = self.__gen_pair_list_from_two_list
        else:
            coupled_pair_path_list = self.coupled_pair_path_list
            pair_path_list = coupled_pair_path_list
            gen_pair_list_func = self.__gen_pair_list_with_coupled_list
        extra_info_pair_path_list =self.extra_info_pair_path_list
        num_pair = len(pair_path_list)
        if self.id_sess_dic is None:
            sub_folder_dic, id_sess_dic = divide_sess_set(self.output_path, num_pair,self.divided_ratio)
        else:
            sub_folder_dic = {x: os.path.join(self.output_path, x) for x in ['train', 'val', 'test', 'debug']}
            id_sess_dic = self.id_sess_dic

        sub_pair_sess_dic = {sess: pair_path_list[id_sess_dic[sess]] for sess in id_sess_dic}
        if self.has_extra_info:
            sub_extra_info_pair_sess_dic = {sess: extra_info_pair_path_list[id_sess_dic[sess]] for sess in id_sess_dic}
        else:
            sub_extra_info_pair_sess_dic = {sess: None for sess in id_sess_dic}
        sess_ratio = {'train': self.divided_ratio[0], 'val': self.divided_ratio[1], 'test': self.divided_ratio[2],
                     'debug': self.divided_ratio[1]}
        if self.max_train_pairs > -1:
            sub_pair_sess_dic['train'] = sub_pair_sess_dic['train'][:self.max_train_pairs]
        pair_list_dic = {sess: gen_pair_list_func(sub_pair_sess_dic[sess], sub_extra_info_pair_sess_dic[sess],
                        int(self.max_total_pairs * sess_ratio[sess]) if self.max_total_pairs > 0 else -1)
                        for sess in sesses}
        divided_path_and_name_dic = self.__gen_path_and_name_dic(pair_list_dic)
        return (sub_folder_dic, divided_path_and_name_dic)


    def save_pair_to_txt(self, info_dict=None):
        """
        the output txt file only contains the source path and the target path
        1. if the source and the target obj do not have extra info,
        then each line refers to the source_obj_path and the  target_obj_path
        2. if the source and the target obj include extra info
        the each line refers to a source_txt and a target txt saved under output_path/info
         this additional txt file  records the path of obj_path, and paths of extra info

        :param info:
        :return:
        """
        sub_folder_dic, divided_path_and_name_dic = info_dict
        if self.has_extra_info:
            saving_pair_info(sub_folder_dic, divided_path_and_name_dic)
        else:
            saving_pair_info_with_extra_info(sub_folder_dic, divided_path_and_name_dic)





