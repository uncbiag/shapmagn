from shapmagn.experiments.datasets.lung.lung_data_analysis import *
from shapmagn.global_variable import *
from shapmagn.utils.utils import get_grid_wrap_points
from shapmagn.shape.point_sampler import uniform_sampler, grid_sampler
from shapmagn.utils.module_parameters import ParameterDict


class PointAug(object):
    def __init__(self, aug_settings):
        self.remove_random_points_by_ratio = aug_settings["remove_random_points_by_ratio"]
        self.add_random_noise_by_ratio = aug_settings["add_random_noise_by_ratio"]
        self.random_noise_raidus = aug_settings["random_noise_raidus"]
        self.normalize_weights = aug_settings["normalize_weights"]

    def remove_random_points(self, points, point_weights, index):
        npoints = points.shape[0]
        nsample = int((1 - self.remove_random_points_by_ratio) * npoints)
        sampler = uniform_sampler(nsample)
        sampling_points, sampling_weights, sampling_index = sampler(points, point_weights)
        return sampling_points, sampling_weights, [index[sind] for sind in sampling_index]

    def add_noises_around_points(self, points, point_weights, index=None):
        npoints, D = points.shape[0], points.shape[-1]
        nnoise = int(self.add_random_noise_by_ratio*npoints)
        index = np.random.choice(list(range(npoints)), nnoise, replace=False)
        noise_disp = torch.ones(nnoise,3).uniform_(-1,1)*self.random_noise_raidus
        noise = points[index] + noise_disp
        points = torch.cat([points, noise],0)
        weights = torch.cat([point_weights,point_weights[index]],0)
        added_index = list(range(npoints+nnoise))
        return points, weights, added_index

    def __call__(self,points, point_weights):
        new_points, new_weights, new_index = points, point_weights, list(range(points.shape[0]))
        if self.remove_random_points_by_ratio!=0:
           new_points, new_weights, new_index = self.remove_random_points(new_points, new_weights, new_index)
        if self.add_random_noise_by_ratio!=0:
           new_points, new_weights, new_index = self.add_noises_around_points(new_points, new_weights, new_index)
        if self.normalize_weights:
            new_points = new_points*(point_weights.sum()/(new_weights.sum()))
        return new_points, new_weights, new_index


class SplineAug(object):
    """
    deform the point cloud via spline deform
    for the grid deformation the isotropic deformation should be used
    for the sampling deformation, either isotropic or anistropic deformation can be used
    for both deformation the nadwat interpolation is used
    :param deform_settings:
    :return:
    """

    def __init__(self,aug_settings):
        super(SplineAug,self).__init__()
        self.aug_settings = aug_settings
        self.do_grid_aug = aug_settings["do_grid_aug"]
        self.do_sampling_aug = aug_settings["do_sampling_aug"]
        grid_aug_settings = self.aug_settings["grid_spline_aug"]
        sampling_aug_settings = self.aug_settings["sampling_spline_aug"]
        grid_spline_kernel_obj = grid_aug_settings[("grid_spline_kernel_obj","","grid spline kernel object")]
        sampling_spline_kernel_obj = sampling_aug_settings[("sampling_spline_kernel_obj","","sampling spline kernel object")]
        self.grid_spline_kernel = obj_factory(grid_spline_kernel_obj) if grid_spline_kernel_obj else None
        self.sampling_spline_kernel = obj_factory(sampling_spline_kernel_obj)  if sampling_spline_kernel_obj else None
        self.plot = aug_settings["plot"]


    def grid_spline_deform(self,points,point_weights):
        grid_aug_settings = self.aug_settings["grid_spline_aug"]
        grid_spacing = grid_aug_settings["grid_spacing"]
        scale = grid_aug_settings["disp_scale"]

        # grid_control_points, _ = get_grid_wrap_points(points, np.array([grid_spacing]*3).astype(np.float32))
        # grid_control_disp = torch.ones_like(grid_control_points).uniform_(-1,1)*scale
        # ngrids = grid_control_points.shape[0]
        # grid_control_weights = torch.ones(ngrids, 1).to(points.device) / ngrids

        sampler = grid_sampler(grid_spacing)
        grid_control_points,grid_control_weights, _ = sampler(points, point_weights)
        grid_control_disp = torch.ones_like(grid_control_points).uniform_(-1,1)*scale

        points_disp = self.grid_spline_kernel(points[None], grid_control_points[None], grid_control_disp[None],grid_control_weights[None])
        deformed_points =points + points_disp[0]
        return deformed_points, point_weights

    def sampling_spline_deform(self,points, point_weights):
        sampling_aug_settings = self.aug_settings["sampling_spline_aug"]
        num_sample = sampling_aug_settings["num_sample"]
        scale = sampling_aug_settings["disp_scale"]
        sampler = uniform_sampler(num_sample)
        sampling_control_points,sampling_control_weights, _ = sampler(points, point_weights)
        sampling_control_disp = torch.ones_like(sampling_control_points).uniform_(-1,1)*scale
        points_disp = self.sampling_spline_kernel(points[None], sampling_control_points[None], sampling_control_disp[None],sampling_control_weights[None])
        deformed_points =points + points_disp[0]
        return deformed_points, point_weights



    def visualize(self, points, deformed_points, point_weights=None, deformed_point_weights=None):
        from shapmagn.utils.visualizer import visualize_point_pair_overlap
        from shapmagn.experiments.datasets.lung.lung_data_analysis import flowed_weight_transform, \
            target_weight_transform
        # visualize_point_pair_overlap(points, deformed_points,
        #                              flowed_weight_transform(point_weights, True),
        #                              target_weight_transform(deformed_point_weights, True),
        #                              title1="original", title2="deformed", rgb_on=False)
        visualize_point_pair_overlap(points, deformed_points,
                                     point_weights,
                                     deformed_point_weights,
                                     title1="original", title2="deformed", rgb_on=False)

    def __call__(self,points, point_weights):
        """

        :param points: torch.tensor  NxD
        :param point_weights: torch.tensor Nx1
        :return:
        """
        deformed_points = points
        deformed_weights = point_weights
        if self.do_sampling_aug:
            deformed_points, deformed_weights = self.sampling_spline_deform(deformed_points,deformed_weights)
        if self.plot:
            self.visualize(points,deformed_points,point_weights,deformed_weights)
        if self.do_grid_aug:
            deformed_points, deformed_weights = self.grid_spline_deform(deformed_points,deformed_weights)
        if self.plot:
            self.visualize(points,deformed_points,point_weights,deformed_weights)
        return deformed_points, deformed_weights




if __name__ == "__main__":
    assert shape_type == "pointcloud", "set shape_type = 'pointcloud'  in global_variable.py"
    device = torch.device("cpu") # cuda:0  cpu
    reader_obj = "lung_dataset_utils.lung_reader()"
    scale = -1  # an estimation of the physical diameter of the lung, set -1 for auto rescaling   #[99.90687, 65.66011, 78.61013]
    normalizer_obj = "lung_dataset_utils.lung_normalizer(scale={})".format(scale)
    sampler_obj = "lung_dataset_utils.lung_sampler(method='voxelgrid',scale=0.001)"
    use_local_mount = True
    remote_mount_transfer = lambda x: x.replace("/playpen-raid1", "/home/zyshen/remote/llr11_mount")
    path_transfer = (lambda x: remote_mount_transfer(x))if use_local_mount else (lambda x: x)

    dataset_json_path = "/home/zyshen/remote/llr11_mount/zyshen/data/point_cloud_expri/train/pair_data.json" #
    dataset_json_path = path_transfer(dataset_json_path)
    pair_name_list, pair_info_list = read_json_into_list(dataset_json_path)
    pair_path_list = [[pair_info["source"]["data_path"], pair_info["target"]["data_path"]] for pair_info in
                      pair_info_list]
    pair_id = 3
    pair_path = pair_path_list[pair_id]
    pair_path = [path_transfer(path) for path in pair_path]
    get_obj_func = get_obj(reader_obj, normalizer_obj, sampler_obj, device,expand_bch_dim=True)
    source, source_interval = get_obj_func(pair_path[0])
    target, target_interval = get_obj_func(pair_path[1])
    source_points, source_weights = source["points"], source["weights"]

    # set deformation
    aug_settings = ParameterDict()
    aug_settings["do_sampling_aug"] = True
    aug_settings["do_grid_aug"] = True
    aug_settings["plot"] = True

    sampling_spline_aug = aug_settings[("sampling_spline_aug",{},"settings for uniform sampling based spline augmentation")]
    sampling_spline_aug["num_sample"] = 1000
    sampling_spline_aug["disp_scale"] = 0.05
    kernel_scale = 0.12
    spline_param = "cov_sigma_scale=0.02,aniso_kernel_scale={},eigenvalue_min=0.3,iter_twice=True, fixed=False, leaf_decay=False, is_interp=True".format(kernel_scale)
    sampling_spline_aug['sampling_spline_kernel_obj']="point_interpolator.NadWatAnisoSpline(exp_order=2,{})".format(spline_param)


    grid_spline_aug = aug_settings[("grid_spline_aug",{},"settings for grid sampling based spline augmentation")]
    grid_spline_aug["grid_spacing"] = 0.9
    grid_spline_aug["disp_scale"] = 0.09
    kernel_scale = 0.1
    grid_spline_aug["grid_spline_kernel_obj"] = "point_interpolator.NadWatIsoSpline(kernel_scale={}, exp_order=2)".format(kernel_scale)

    data_augmentation = SplineAug(aug_settings)
    data_augmentation(source_points[0],source_weights[0])
