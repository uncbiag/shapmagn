from glob import glob
import os
from shapmagn.datasets.prepare_data import CustomDataSet
data_folder_path = "/playpen-raid1/Data/UNC_vesselParticles"
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