"""
data reader for the toys
given a file path, the reader will return a dict
{"points":Nx3, "pointfea": NxFeaDim, "weights":Nx1}
"""
from shapmagn.datasets.vtk_utils import read_vtk

def toy_reader():
    """
    :return:
    """
    reader = read_vtk
    def read(file_info):
        path = file_info["data_path"]
        raw_data_dict = reader(path)
        data_dict = {}
        data_dict["points"] = raw_data_dict["points"]
        data_dict["faces"] = raw_data_dict["faces"]
        return data_dict
    return read



def toy_sampler():
    """
    :param args:
    :return:
    """
    def do_nothing(data_dict):
        return data_dict
    return do_nothing


def toy_normalizer():
    """
    :return:
    """

    def do_nothing(data_dict):
        return data_dict

    return do_nothing



if __name__ == "__main__":
    from shapmagn.utils.obj_factory import obj_factory
    reader_obj = "toy_dataset_utils.toy_reader()"
    sampler_obj = "toy_dataset_utils.toy_sampler()"
    normalizer_obj = "toy_dataset_utils.toy_normalizer()"
    reader = obj_factory(reader_obj)
    normalizer = obj_factory(normalizer_obj)
    sampler = obj_factory(sampler_obj)
    file_path = "/playpen-raid1/zyshen/proj/shapmagn/shapmagn/datasets/toy/toy_synth/divide_3d_sphere_level1.vtk"
    file_info = {"name":file_path,"data_path":file_path}
    raw_data_dict  = reader(file_info)
    normalized_data_dict = normalizer(raw_data_dict)
    sampled_data_dict = sampler(normalized_data_dict)

