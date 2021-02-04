import os, sys
sys.path.insert(0, os.path.abspath('../../../..'))
from copy import deepcopy
import numpy as np
import torch
import matplotlib.pyplot as plt
from shapmagn.global_variable import Shape, shape_type
from shapmagn.datasets.data_utils import read_json_into_list, get_obj
from shapmagn.shape.shape_pair_utils import create_shape_pair
from shapmagn.utils.obj_factory import obj_factory
from shapmagn.utils.visualizer import visualize_point_fea, visualize_point_pair, visualize_multi_point
from shapmagn.utils.local_feature_extractor import *



def get_pair(source_path, target_path):

    get_obj_func = get_obj(reader_obj, normalizer_obj, sampler_obj, device,expand_bch_dim=True)
    source_obj, source_interval = get_obj_func(source_path)
    target_obj, target_interval = get_obj_func(target_path)
    return source_obj, target_obj


def plot_pair_weight_distribution(source_weight, target_weight, use_log=False):
    plt.style.use('bmh')
    fig, ax = plt.subplots()
    source_weight = np.log(source_weight) if use_log else source_weight
    target_weight = np.log(target_weight) if use_log else target_weight
    ax.hist(source_weight, bins=50, density=1, histtype='stepfilled', alpha=0.5)
    ax.hist(target_weight, bins=50, density=1, histtype='stepfilled', alpha=0.5)
    ax.set_title('weight' if not use_log else "log_weight")
    plt.show()
    plt.clf()

def get_half_lung(lung, normalize_weight=False):
    weights = lung.weights.detach()
    points = lung.points.detach()
    pos_filter = points[...,0] < 0
    points = points[pos_filter][None]
    weights = weights[pos_filter][None]
    weights = weights
    weights = weights/weights.sum() if normalize_weight else weights
    half_lung = Shape()
    half_lung.set_data(points=points, weights=weights)
    return half_lung


def get_key_vessel(lung,thre=2e-05):
    weights = lung.weights.detach()
    points = lung.points.detach()
    mask = (lung.weights>thre)[...,0]
    weights = weights[mask][None]
    points = points[mask][None]
    key_lung = Shape()
    key_lung.set_data(points=points,weights=weights)
    return key_lung




def source_weight_transform(weights,compute_on_half_lung=False):
    weights = weights * 1
    weights_cp = deepcopy(weights)
    thre = 2e-05
    thre = thre if not compute_on_half_lung else thre*2
    weights[weights_cp < thre] = 1e-7
    return weights


def flowed_weight_transform(weights,compute_on_half_lung=False):
    weights = weights * 1
    weights_cp = deepcopy(weights)
    thre = 2e-05
    thre = thre if not compute_on_half_lung else thre * 2
    weights[weights_cp < thre] = 1e-7
    return weights

def target_weight_transform(weights,compute_on_half_lung=False):
    weights = weights * 1
    weights_cp = deepcopy(weights)
    thre = 1.4e-05
    thre = thre if not compute_on_half_lung else thre * 2
    weights[weights_cp < thre] = 1e-7
    # weights[weights_cp > 1.1e-05] = 1e-7
    return weights


def capture_plotter():
    from shapmagn.utils.visualizer import visualize_point_pair_overlap
    inner_count = 0
    def save(record_path,name_suffix, shape_pair):
        nonlocal  inner_count
        source, flowed, target = shape_pair.source, shape_pair.flowed, shape_pair.target
        if inner_count==0:
            path = os.path.join(record_path,"source_target"+"_"+name_suffix+".png")
            visualize_point_pair_overlap(source.points, target.points,
                                         flowed_weight_transform(flowed.weights, True),
                                         target_weight_transform(target.weights, True),
                                         title1="source", title2="target", rgb_on=False,saving_capture_path=path, show=False)
        path = os.path.join(record_path, "flowed_target" + "_" + name_suffix + ".png")
        visualize_point_pair_overlap(flowed.points, target.points,
                                 flowed_weight_transform(flowed.weights,True),
                                 target_weight_transform(target.weights,True),
                                 title1="flowed",title2="target", rgb_on=False,saving_capture_path=path, show=False)
        inner_count +=1
    return save



def lung_isolated_leaf_clean_up(lung, radius=0.032, principle_weight=None, normalize_weights=True):

    points = lung.points.detach()
    weights = lung.weights.detach()
    mass, dev, cov = compute_local_moments(points, radius=radius)
    eigenvector_main = compute_local_fea_from_moments("eigenvector_main", mass, dev, cov)
    filter = mass[..., 0].squeeze() > 2
    to_remove = ~filter
    print("In the first step, num of points are removed {}, {}".format(torch.sum(to_remove),torch.sum(to_remove)/len(filter) ))
    points_toremove = points[:,to_remove]
    mass_toremove = mass[:,to_remove]
    mass = mass[:, filter]
    points = points[:, filter]
    weights = weights[:, filter]
    eigenvector_main = eigenvector_main[:, filter]
    visualize_point_fea_with_arrow(points, mass, eigenvector_main * 0.01, rgb_on=False)
    visualize_point_overlap(points,points_toremove,mass,mass_toremove,title="cleaned points",point_size=(10,20),rgb_on=False,opacity=('linear',1.0))


    Gamma= compute_anisotropic_gamma_from_points(points, cov_sigma_scale=radius, aniso_kernel_scale=radius, principle_weight=principle_weight)
    mass, dev, cov = compute_aniso_local_moments(points, Gamma)
    eigenvector_main = compute_local_fea_from_moments("eigenvector_main", mass, dev, cov)
    filter = mass[..., 0].squeeze() > 2.5
    to_remove = ~filter
    print("In the second step, num of points are removed {}, {}".format(torch.sum(to_remove), torch.sum(to_remove) / len(filter)))
    points_toremove = points[:, to_remove]
    mass_toremove = mass[:, to_remove]
    mass = mass[:, filter]
    points = points[:, filter]
    weights = weights[:, filter]
    eigenvector_main = eigenvector_main[:, filter]
    visualize_point_fea_with_arrow(points, mass, eigenvector_main * 0.01, rgb_on=False)
    visualize_point_overlap(points,points_toremove,mass,mass_toremove,title="cleaned points",point_size=(10,20),rgb_on=False,opacity=('linear',1.0))


    Gamma = compute_anisotropic_gamma_from_points(points, cov_sigma_scale=radius, aniso_kernel_scale=radius, principle_weight=principle_weight)
    mass, dev, cov = compute_aniso_local_moments(points, Gamma)
    eigenvector_main = compute_local_fea_from_moments("eigenvector_main", mass, dev, cov)
    filter = mass[..., 0].squeeze() > 3
    to_remove = ~filter
    print("In the third step, num of points are removed {}, {}".format(torch.sum(to_remove), torch.sum(to_remove) / len(filter)))
    points_toremove = points[:, to_remove]
    mass_toremove = mass[:, to_remove]
    mass = mass[:,filter]
    points = points[:, filter]
    weights = weights[:, filter]
    eigenvector_main = eigenvector_main[:,filter]
    visualize_point_fea_with_arrow(points, mass,eigenvector_main*0.01,rgb_on=False)
    visualize_point_overlap(points,points_toremove,mass,mass_toremove,title="cleaned points",point_size=(10,20),rgb_on=False,opacity=('linear',1.0))

    cleaned_lung = Shape()
    cleaned_lung.points, cleaned_lung.weights = points, weights/torch.sum(weights) if normalize_weights else weights
    return cleaned_lung

def analysis_large_vessel(source, target, source_weight_transform=source_weight_transform, target_weight_transform=target_weight_transform, title1="source", title2="target"):
    source_points, source_weights,  = source.points.detach().cpu(), source.weights.detach().cpu()
    target_points, target_weights,  = target.points.detach().cpu(), target.weights.detach().cpu()
    plot_pair_weight_distribution(source_weight_transform(source_weights).squeeze().numpy(),
                                  target_weight_transform(target_weights).squeeze().numpy(),
                                  use_log=True)
    visualize_point_pair(source_points, target_points,
                         source_weight_transform(source_weights),
                         target_weight_transform(target_weights),
                         title1=title1, title2=title2, rgb_on=False)

if __name__ == "__main__":
    assert shape_type == "pointcloud", "set shape_type = 'pointcloud'  in global_variable.py"
    device = torch.device("cuda:0") # cuda:0  cpu
    reader_obj = "lung_dataset_utils.lung_reader()"
    scale = 80  # an estimation of the physical diameter of the lung, set -1 for auto rescaling   #[99.90687, 65.66011, 78.61013]
    normalizer_obj = "lung_dataset_utils.lung_normalizer(scale={})".format(scale)
    sampler_obj = "lung_dataset_utils.lung_sampler(method='voxelgrid',scale=0.001)"
    use_local_mount = True
    remote_mount_transfer = lambda x: x.replace("/playpen-raid1", "/home/zyshen/remote/llr11_mount")
    path_transfer = (lambda x: remote_mount_transfer(x))if use_local_mount else (lambda x: x)

    dataset_json_path = "/playpen-raid1/zyshen/data/point_cloud_expri/train/pair_data.json"
    dataset_json_path = path_transfer(dataset_json_path)
    pair_name_list, pair_info_list = read_json_into_list(dataset_json_path)
    pair_path_list = [[pair_info["source"]["data_path"], pair_info["target"]["data_path"]] for pair_info in
                      pair_info_list]
    pair_id = 3
    pair_path = pair_path_list[pair_id]
    pair_path = [path_transfer(path) for path in pair_path]
    source, target = get_pair(*pair_path)
    source_weight, target_weight = source["weights"].squeeze().cpu().numpy(), target["weights"].squeeze().cpu().numpy()
    #plot_pair_weight_distribution(source_weight, target_weight, use_log=True)

    input_data = {"source": source, "target": target}
    source_target_generator = obj_factory("shape_pair_utils.create_source_and_target_shape()")
    source, target = source_target_generator(input_data)
    shape_pair = create_shape_pair(source, target)
    source_half = get_half_lung(source)
    target_half = get_half_lung(target)
    # cleaned_source_half = lung_isolated_leaf_clean_up(source_half,radius=0.032, principle_weight=[2,1,1], normalize_weights=False)
    # visualize_point_pair(source_half.points, cleaned_source_half.points,
    #                      source_weight_transform(source_half.weights),
    #                      source_weight_transform(cleaned_source_half.weights),
    #                      title1="source", title2="cleaned_source", rgb_on=False)
    #
    # plot_pair_weight_distribution(source_weight_transform(source_half.weights).cpu().squeeze().numpy(),
    #                               target_weight_transform(target_half.weights).cpu().squeeze().numpy(),
    #                               use_log=True)

    visualize_point_pair(source_half.points, target_half.points,
                         source_weight_transform(source_half.weights),
                         target_weight_transform(target_half.weights),
                         title1="source", title2="target", rgb_on=False)