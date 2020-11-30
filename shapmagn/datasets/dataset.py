from __future__ import print_function, division
import blosc
import torch
from torch.utils.data import Dataset
from shapmagn.utils.obj_factory import obj_factory
from shapmagn.datasets.data_utils import *
from multiprocessing import *
blosc.set_nthreads(1)
from tqdm import tqdm

class RegistrationDataset(Dataset):
    """registration dataset."""

    def __init__(self, data_root_path, option=None, phase=None):
        """
        the dataloader for registration task, to avoid frequent disk communication, all pairs can be optionally compressed into memory
        :param data_root_path:  string, path to the data
            the data should be preprocessed and saved into txt
        :param phase:  string, 'train'/'val'/ 'test'/ 'debug' ,    debug here means a subset of train data, to check if model is overfitting
        : option:  pars, settings for registration task

        """
        self.phase = phase
        self.data_path = os.path.join(data_root_path, phase)
        self.transform = ToTensor()
        self.aug_data_via_inverse_reg_direction = option[("aug_data_via_inverse_reg_direction",False," aug_data_via_inverse_reg_direction")]
        """ inverse the registration order, i.e the original set is A->B, the new set would be A->B and B->A """
        ind = ['train', 'val', 'test', 'debug'].index(phase)
        max_num_for_loading=option['max_num_for_loading',(-1,-1,-1,-1),"the max number of pairs to be loaded, set -1 if there is no constraint,[max_train, max_val, max_test, max_debug]"]
        self.max_num_for_loading = max_num_for_loading[ind]
        """ the max number of pairs to be loaded into the memory,[max_train, max_val, max_test, max_debug]"""
        self.pair_name_list = []
        self.pair_info_list = []
        self.pair_list = []
        self.get_file_list()
        self.reg_option = option
        self.reader = obj_factory(option[('reader',"","a reader instance")])
        self.sampler = obj_factory(option[('sampler',"","a sampler instance")])
        self.normalizer = obj_factory(option[('normalizer',"", "a normalizer instance")])
        load_training_data_into_memory = option[('load_training_data_into_memory',True, "when train network, load all training sample into memory can relieve disk burden")]
        self.load_into_memory = load_training_data_into_memory if phase == 'train' else False

        if self.load_into_memory:
            self._init_data_pool()

    def get_file_list(self):
        """
        """
        if not os.path.exists(self.data_path):
            raise IOError("Non data loaded")
        self.pair_name_list, self.pair_info_list = read_json_into_list(os.path.join(self.data_path,'pair_data.json'))
        read_num = min(self.max_num_for_loading, len(self.pair_info_list))
        if self.max_num_for_loading>0:
            self.pair_info_list = self.pair_info_list[:read_num]
            self.pair_name_list = self.pair_name_list[:read_num]

        if self.aug_data_via_inverse_reg_direction and self.phase=='train':
            for pair_info in self.pair_info_list:
                pair_info_list_inverse = [[pair_info[1], pair_info[0]]]
                pair_name_list_inverse = [self._inverse_name(name) for name in self.pair_name_list]
                self.pair_info_list += pair_info_list_inverse
                self.pair_name_list += pair_name_list_inverse



    def _init_data_pool(self):
        """

        """
        manager = Manager()
        data_dic = manager.dict()
        data_info_dic = {}
        _pair_name_list = []
        for pair_info in self.pair_info_list:
            source_info, target_info = pair_info["source"], pair_info["target"]
            sname, tname = source_info['name'], target_info['name']
            if sname not in data_info_dic:
                data_info_dic[sname] = source_info
            if tname not in data_info_dic:
                data_info_dic[tname] = target_info
            _pair_name_list.append([sname, tname])

        #  multi process
        num_of_workers = 12
        num_of_workers = num_of_workers if len(_pair_name_list) > 12 else 2
        dict_splits = split_dict(data_info_dic, num_of_workers)
        procs = []
        for i in range(num_of_workers):
            p = Process(target=self._data_into_zipnp,args=(dict_splits[i], data_dic,))
            p.start()
            print("pid:{} start:".format(p.pid))
            procs.append(p)
        for p in procs:
            p.join()
        print("the loading phase finished, total {} data have been loaded".format(len(data_dic)))
        data_dic=dict(data_dic)

        # organize data into pair list
        for pair_name in _pair_name_list:
            sname = pair_name[0]
            tname = pair_name[1]
            self.pair_list.append([data_dic[sname],data_dic[tname]])



    def _preprocess_data(self, file_info):
        """
        preprocess the data :
        1. read the data into dict, and return the shape_type (pointcloud, polyline, surfacemesh)
        2. sampling on the data dict
        3. normalize the data dict
        :param path: data_path
        :return: data_dict, shape_type
        """
        case_dict = self.reader(file_info)
        case_dict = self.sampler(case_dict)
        case_dict = self.normalizer(case_dict)
        return case_dict



    def _data_into_zipnp(self,data_path_dic, data_dict):
        """
        compress the data into zip to save memory
        :param data_path_dic:
        :param data_dict:
        :return:
        """
        for fn in tqdm(data_path_dic):
            case_dict = self._preprocess_data(data_path_dic[fn])
            zip_fn = lambda x: blosc.pack_array(x)
            data_dict[fn] = {key: zip_fn(case_dict[key]) for key in case_dict}




    def _inverse_name(self,name):
        """get the name of the inversed registration pair"""
        name = name+'_inverse'
        return name



    def __len__(self):
        # to make the epoch size always meet the setting, we scale the dataset when training dataset size is too small
        return len(self.pair_name_list)*500 if len(self.pair_name_list)<200 and self.phase=='train' else len(self.pair_name_list)



    def __getitem__(self, idx):
        """
        get pair data_dict
        {"source": source_dict, "target":target_dict,
                           "shape_type":shape_type, "pair_name":pair_name,
                           "source_info":source_info, "target_info":target_info}

        :param idx: id of the items
        :return: pair_data_dict, pair_name

        """
        # print(idx)
        idx = idx %len(self.pair_name_list)
        pair_info = self.pair_info_list[idx]
        pair_name = self.pair_name_list[idx]
        source_info, target_info = pair_info["source"], pair_info["target"]
        if not self.load_into_memory:
            source_dict = self._preprocess_data(source_info)
            target_dict = self._preprocess_data(target_info)
        else:
            zip_source_dict, zip_target_dict = self.pair_list[idx]
            unzip_fn = lambda x: blosc.unpack_array(x)
            unzip_shape_fn = lambda shape_dict: {key: unzip_fn(fea) for key, fea in shape_dict.items()}
            source_dict = unzip_shape_fn(zip_source_dict)
            target_dict = unzip_shape_fn(zip_target_dict)

        if self.transform:
            source_dict = {key:self.transform(fea) for key,fea in source_dict.items()}
            target_dict = {key:self.transform(fea) for key,fea in target_dict.items()}

        sample = {"source": source_dict, "target":target_dict,
                   "pair_name":pair_name,
                  "source_info":source_info, "target_info":target_info}
        return sample, pair_name



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        n_tensor = torch.from_numpy(sample)
        return n_tensor


if __name__ == "__main__":
    from shapmagn.utils.module_parameters import ParameterDict
    data_path = "/playpen-raid1/zyshen/data/lung_pointcloud/debugging"
    reader_obj = "lung_dataset_utils.lung_reader()"
    sampler_obj = "lung_dataset_utils.lung_sampler(num_sample=1000, method='uniform')"
    # sampler_obj = "lung_dataset_utils.lung_sampler(num_sample=1000, method='voxelgrid',scale=5)"
    normalizer_obj = "lung_dataset_utils.lung_normalizer(scale=[100,100,100],shift=[50,50,50])"
    data_opt = ParameterDict()
    data_opt["reader"] = reader_obj
    data_opt["sampler"] = sampler_obj
    data_opt["normalizer"] = normalizer_obj
    data_opt["max_num_for_loading"] = [20, 3, 3, 1]
    dataset = RegistrationDataset(data_path, option=data_opt, phase="train")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=3,
                                shuffle=True, num_workers=3)
    for data in dataloader:
        print("the name of the data is {}".format(data[1]))