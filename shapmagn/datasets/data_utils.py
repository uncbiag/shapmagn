from os import listdir
from os.path import isfile, join
import json
import os
import numpy as np
import random
import torch
from shapmagn.utils.obj_factory import obj_factory

def list_dic(path):
    return [ dic for dic in listdir(path) if not isfile(join(path,dic))]


def split_dict(dict_to_split,split_num):
    index_list = list(range(len(dict_to_split)))
    index_split = np.array_split(np.array(index_list),split_num)
    split_dict=[]
    dict_to_split_items = list(dict_to_split.items())
    for i in range(split_num):
        dj=dict(dict_to_split_items[index_split[i][0]:index_split[i][-1]+1])
        split_dict.append(dj)
    return split_dict

def make_dir(path):
    is_exist = os.path.exists(path)
    if not is_exist:
        os.makedirs(path)
    return is_exist


def str_concat(lists,linker='_'):
    from functools import reduce
    str_concated = reduce((lambda x,y:str(x)+linker+str(y)), lists)
    return str_concated

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)


def divide_sess_set(root_path, pair_num,ratio):
    """
    divide the dataset into root_path/train, root_path/val, root_path/test, root_path/debug
    :param root_path: the root path for saving the task_dataset
    :param pair_num: num of pair
    :param ratio: tuple of (train_ratio, val_ratio, test_ratio, debug_ratio) from all the pairs
    :return:  full path of each file

    """
    train_ratio = ratio[0]
    val_ratio = ratio[1]
    sub_folder_dic = {x:os.path.join(root_path,x) for x in ['train', 'val', 'test','debug']}
    # debug details maybe added later
    make_dir(os.path.join(root_path,'debug'))
    nt = [make_dir(sub_folder_dic[key]) for key in sub_folder_dic]

    # if sum(nt):
    #     raise ValueError("the data has already exist, due to randomly assignment schedule, the program block\n" \
    #                      "manually delete the folder to reprepare the data")
    train_num = int(train_ratio * pair_num)
    val_num = int(val_ratio*pair_num)
    file_id_dic={}
    file_id_dic['train'] = list(range(train_num))
    file_id_dic['val'] = list(range(train_num, train_num+val_num))
    file_id_dic['test'] = list(range(train_num+val_num,pair_num))
    file_id_dic['debug'] = list(range(train_num))
    return sub_folder_dic, file_id_dic




def get_divided_dic(file_id_dic, pair_path_list, pair_name_list):
    """
    get the set dict of the image pair path and pair name

    :param file_id_dic: dict of set index, {'train':[1,2,3,..100], 'val':[101,...120]......}
    :param pair_path_list: list of pair_path, [[s1,t1], [s2,t2],.....]
    :param pair_name_list: list of fnmae [s1_t1, s2_t2,....]
    :return: divided_path_dic {'pair_path_list':{'train': [[s1,t1],[s2,t2]..],'val':[..],...}, 'pair_name_list':{'train':[s1_t1, s2_t2,...],'val':[..],..}}
    """
    divided_path_dic = {}
    sesses = ['train','val','test','debug']
    divided_path_dic['pair_path_list'] ={sess:[pair_path_list[idx] for idx in file_id_dic[sess]] for sess in sesses}
    divided_path_dic['pair_name_list'] ={sess:[pair_name_list[idx] for idx in file_id_dic[sess]] for sess in sesses}
    return divided_path_dic




def get_extra_info_path_list(file_path_list, extra_info_folder_path=None, replacer = ('','')):
    """
    get the extra_info path
    :param file_path_list: the path list with given file path and their replacer
    :param extra_info_folder_path: the folder path of the extra_info
    :return:
    """
    replace_fn = lambda x: x.replace(replacer[0], replacer[1])

    if extra_info_folder_path is not None:
        extra_info_path_list =  [os.path.join(extra_info_folder_path, replace_fn(os.path.split(file_path)[1])) for file_path in file_path_list]
    else:
        # assume the extra_info is in the same folder as given files
        extra_info_path_list = [ replace_fn(file_path) for file_path in file_path_list]
    return extra_info_path_list



def saving_pair_info(sub_folder_dic, pair_list_dic):
    for sess, sub_folder_path in sub_folder_dic.items():
        pair_sess_list = pair_list_dic[sess]
        pair_name_list = [str_concat([pair[0]["name"],pair[1]["name"]]) for pair in pair_sess_list]
        pair_path_list = [[pair[0]["data_path"],pair[1]["data_path"]] for pair in pair_sess_list]
        output_dict = {pair_name: {"source":pair[0],"target":pair[1]} for pair_name, pair in zip(pair_name_list,pair_sess_list)}
        save_json(os.path.join(sub_folder_path,"pair_data.json"), output_dict)
        write_list_into_txt(os.path.join(sub_folder_path,"pair_name_list.txt"), pair_name_list)
        write_list_into_txt(os.path.join(sub_folder_path,"pair_path_list.txt"), pair_path_list)



def read_json_into_list(file_path):
    with open(file_path) as f:
        data_dict = json.load(f)
    data_name_list = [name for name in data_dict]
    data_list = [data_dict[name] for name in data_name_list]
    return data_name_list, data_list



def write_list_into_txt(file_path, list_to_write):
    """
    write the list into txt,  each elem refers to a line
    if elem is also a list, then each sub elem is separated by the space

    :param file_path: the file path to write in
    :param list_to_write: the list to refer
    :return: None
    """
    with open(file_path, 'w') as f:
        if len(list_to_write):
            if isinstance(list_to_write[0],(float, int, str)):
                f.write("\n".join(list_to_write))
            elif isinstance(list_to_write[0],(list, tuple)):
                new_list = ["     ".join(sub_list) for sub_list in list_to_write]
                f.write("\n".join(new_list))
            else:
                raise(ValueError,"not implemented yet")

def read_txt_into_list(file_path):
    """
    read from the file, returns a nested list,  the outer refers to line, the inner refers to the line item
    if item is a string "None", it would be filtered and not considered
    :param file_path: the file path to read
    :return: list of list
    """
    assert os.path.isfile(file_path), "the file {} doesnt exist".format(file_path)
    import re
    lists= []
    with open(file_path,'r') as f:
        content = f.read().splitlines()
        if len(content)>0:
            lists= [[x if x!='None'else None for x in re.compile('\s*[,|\s+]\s*').split(line)] for line in content]
            lists = [list(filter(lambda x: x is not None, items)) for items in lists]
        lists = [item[0] if len(item) == 1 else item for item in lists]
    return lists





def read_fname_list_from_pair_fname_txt(file_path,return_separate_name=False):
    """
    the txt file may has two type
    1) 1 item per line,  that is the pair_name
    2) 3 item per line, that is the pair_name  moving_name  target_name
    :param file_path:
    :param return_separate_name:
    :return:
    """
    fname_list = read_txt_into_list(file_path)
    if len(fname_list) and isinstance(fname_list[0],list):
        if return_separate_name:
            return fname_list
        else:
            return [fname[0] for fname in fname_list]
    else:
        return fname_list

def get_file_name(file_path,last_ocur=True):
    if not last_ocur:
        name= os.path.split(file_path)[1].split('.')[0]
    else:
        name = os.path.split(file_path)[1].rsplit('.',1)[0]
    return name

def generate_pair_name(pair_path,return_separate_name=False):
    """
    get the filename and drop the file type, if source and target name are the same,
    then iteratively check their parent folder names util they are different
    :param pair_path:
    :param return_separate_name:
    :return:
    """
    source_path, target_path = pair_path
    f = lambda x: os.path.split(x)
    while True:
        s = get_file_name(f(source_path)[-1])
        t = get_file_name(f(target_path)[-1])
        if s !=t:
            break
        else:
            source_path, target_path = f(source_path)[0],f(target_path)[0]
    pair_name = s+"_"+t
    if not return_separate_name:
        return pair_name
    else:
        return pair_name, s, t


def compute_interval(vertices):
    if len(vertices)<2000:
        vert_i  = vertices[:,None]
        vert_j  = vertices[None]
        vert_dist = ((vert_i-vert_j)**2).sum(-1)
        vert_dist = np.sqrt(vert_dist)
        min_interval = np.min(vert_dist[np.where(vert_dist>0)])
        print("the min interval is {}".format(min_interval))
        return min_interval
    else:
        sampled_index = random.sample(list(range(len(vertices)-1)), 2000)
        sampled_index_plus = [index+1 for index in sampled_index]
        vertices_sampled = vertices[sampled_index]
        vertices_sampled_plus = vertices[sampled_index_plus]
        vert_dist = np.sqrt(((vertices_sampled-vertices_sampled_plus)**2).sum(-1))
        sampled_min_interval = np.min(vert_dist[np.where(vert_dist>0)])
        print("the min interval is {}".format(sampled_min_interval))
        return sampled_min_interval






def get_obj(reader_obj,normalizer_obj=None,sampler_obj=None,pair_postprocess_obj=None, device=None, expand_bch_dim=True):
    def _get_obj(file_path):
        name = get_file_name(file_path)
        file_info = {"name":name,"data_path":file_path}
        reader = obj_factory(reader_obj)
        raw_data_dict  = reader(file_info)
        normalizer = obj_factory(normalizer_obj) if normalizer_obj else None
        data_dict = normalizer(raw_data_dict) if normalizer_obj else raw_data_dict
        min_interval = compute_interval(data_dict["points"])
        pair_postprocess =obj_factory(pair_postprocess_obj) if pair_postprocess_obj else None
        data_dict = pair_postprocess(data_dict) if pair_postprocess_obj else data_dict
        sampler = obj_factory(sampler_obj) if sampler_obj else None
        data_dict = sampler(data_dict) if sampler_obj else data_dict
        obj = {key: torch.from_numpy(fea).to(device) for key, fea in data_dict.items()}
        if expand_bch_dim:
            obj= {key: fea[None] for key, fea in obj.items()}
        return obj, min_interval
    return _get_obj








