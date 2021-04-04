from glob import glob
import os

from shapmagn.datasets.data_utils import save_json
from shapmagn.datasets.prepare_data import CustomDataSet
use_dirlab_as_val = True
data_folder_path = "/playpen-raid1/Data/UNC_vesselParticles"
dirlab_folder_path = "/playpen-raid1/Data/DIRLABVascular"
insp_key = "*_INSP_STD_*"
insp_path_list= glob(os.path.join(data_folder_path,insp_key))
exp_path_list = [path.replace("INSP","EXP") for path in insp_path_list]
for exp_path in exp_path_list:
    assert os.path.isfile(exp_path),"the file {} is not exist".format(exp_path)
print("num of {} pair detected".format(len(insp_path_list)))
file_name_list = [os.path.split(path)[-1].split("_")[0] for path in insp_path_list]
insp_name_list = [name+"_insp" for name in file_name_list]
exp_name_list = [name+"_exp" for name in file_name_list]
insp_list = [{"name":insp_name, "data_path":insp_path}
             for insp_name, insp_path in zip(insp_name_list, insp_path_list)]
exp_list = [{"name":exp_name, "data_path":exp_path}
             for exp_name, exp_path in zip(exp_name_list, exp_path_list)]
dataset = CustomDataSet()
dataset.set_output_path("/playpen-raid1/zyshen/data/lung_pointcloud/debugging")
dataset.set_coupled_pair_list([exp_list,insp_list])
dataset.set_divided_ratio((0.6,0.1,0.3))
dataset.prepare_data()

if use_dirlab_as_val:
    dirlab_list=[
    "12042G",
    "12105E",
    "12109M",
    "12239Z",
    "12829U",
    "13216S",
    "13528L",
    #"13671Q",
    "13998W",
    "17441T"]
    source_name_list = [name+"_EXP_STD_USD_COPD" for name in dirlab_list]
    source_path_list = [os.path.join(dirlab_folder_path,name+"_wholeLungVesselParticles.vtk") for name in source_name_list]
    target_name_list = [name+"_INSP_STD_USD_COPD" for name in dirlab_list]
    target_path_list = [os.path.join(dirlab_folder_path,name+"_wholeLungVesselParticles.vtk") for name in target_name_list]

    source_list = [{"name":name, "data_path":os.path.abspath(path)}
                 for name, path in zip(source_name_list, source_path_list)]
    target_list = [{"name":name, "data_path":os.path.abspath(path)}
                 for name, path in zip(target_name_list, target_path_list)]
    output_folder = "/playpen-raid1/zyshen/data/lung_expri/dirlab_val"
    os.makedirs(output_folder,exist_ok=True)
    output_dict = {pair_name: {"source": source,"target":target} for pair_name, source, target in zip(dirlab_list,source_list,target_list)}
    save_json(os.path.join(output_folder, "pair_data.json"), output_dict)

# import json
# from shapmagn.datasets.data_utils import save_json
# for phase in ["train","val","debug","test"]:
#     with open('/playpen-raid1/zyshen/data/lung_pointcloud/debugging/{}/pair_data.json'.format(phase)) as f:
#         pair_info_dict = json.load(f)
#     output_dict = {pair_info_dict[pair_name]["source"]["name"]:pair_info_dict[pair_name]["source"]["data_path"] for pair_name in pair_info_dict}
#     target_dict = {pair_info_dict[pair_name]["target"]["name"]:pair_info_dict[pair_name]["target"]["data_path"] for pair_name in pair_info_dict}
#     output_dict.update(target_dict)
#     save_json("/playpen-raid1/zyshen/data/lung_pointcloud/debugging/{}/single_data.json".format(phase), output_dict)




