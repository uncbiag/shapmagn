import os
from shapmagn.datasets.data_utils import save_json

data_folder_path = "/playpen-raid1/zyshen/debug/body_registration"
source_path_list = [os.path.join(data_folder_path, "source.ply")]
target_path_list = [os.path.join(data_folder_path, "target.ply")]
for path in target_path_list:
    assert os.path.isfile(path), "the file {} is not exist".format(path)
pair_name_list = ["source_target"]
pair_sess_list = ["test"]
source_name_list = ["source"]
target_name_list = ["target"]
source_list = [
    {"name": name, "data_path": os.path.abspath(path)}
    for name, path in zip(source_name_list, source_path_list)
]
target_list = [
    {"name": name, "data_path": os.path.abspath(path)}
    for name, path in zip(target_name_list, target_path_list)
]
output_folder = "/playpen-raid1/zyshen/data/unc_body"
os.makedirs(output_folder, exist_ok=True)
output_dict = {
    pair_name: {"source": source, "target": target}
    for pair_name, source, target in zip(pair_name_list, source_list, target_list)
}
save_json(os.path.join(output_folder, "pair_data.json"), output_dict)
