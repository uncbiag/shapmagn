import torch
import numpy as np
from shapmagn.utils.obj_factory import obj_factory
from shapmagn.global_variable import Shape
from shapmagn.utils.utils import get_grid_wrap_points
from shapmagn.shape.point_sampler import point_uniform_sampler, point_grid_sampler,uniform_sampler, grid_sampler



def visualize(points, deformed_points, point_weights=None, deformed_point_weights=None):
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




class PointAug(object):
    def __init__(self, aug_settings):
        self.remove_random_points = aug_settings["remove_random_points"]
        self.add_random_point_noise = aug_settings["add_random_point_noise"]
        self.add_random_weight_noise = aug_settings["add_random_weight_noise"]
        self.remove_random_points_by_ratio = aug_settings["remove_random_points_by_ratio"]
        self.add_random_point_noise_by_ratio = aug_settings["add_random_point_noise_by_ratio"]
        self.random_noise_raidus = aug_settings["random_noise_raidus"]
        self.normalize_weights = aug_settings["normalize_weights"]
        self.plot = aug_settings["plot"]

    def remove_random_points(self, points, point_weights, index):
        npoints = points.shape[0]
        nsample = int((1 - self.remove_random_points_by_ratio) * npoints)
        sampler = uniform_sampler(nsample,fixed_random_seed=False,sampled_by_weight=True)
        sampling_points, sampling_weights, sampling_index = sampler(points, point_weights)
        return sampling_points, sampling_weights, index[sampling_index]

    def add_noises_around_points(self, points, point_weights, index=None):
        npoints, D = points.shape[0], points.shape[-1]
        nnoise = int(self.add_random_point_noise_by_ratio*npoints)
        rand_index = np.random.choice(list(range(npoints)), nnoise, replace=False)
        noise_disp = torch.ones(nnoise,3, device=points.device).uniform_(-1,1)*self.random_noise_raidus
        noise = points[rand_index] + noise_disp
        points = torch.cat([points, noise],0)
        weights = torch.cat([point_weights,point_weights[rand_index]],0)
        added_index = torch.tensor(list(range(npoints,npoints+nnoise))).to(points.device)
        return points, weights, torch.cat([index,added_index])

    def add_random_noise_to_weights(self, points, point_weights, index=None):
        noise_std = (torch.min(point_weights) / 5).item()
        weights_noise = torch.ones_like(point_weights).normal_(0, noise_std)
        point_weights = point_weights+ weights_noise
        return points, point_weights, index

    def __call__(self,points, point_weights):
        N, D = points.shape[0], points.shape[1]
        device = points.device
        new_points, new_weights, new_index = points, point_weights, torch.tensor(list(range(N))).to(device)
        if self.remove_random_points and self.remove_random_points_by_ratio!=0:
            new_points, new_weights, new_index = self.remove_random_points(new_points, new_weights, new_index)
        if self.add_random_point_noise and  self.add_random_point_noise_by_ratio!=0:
            new_points, new_weights, new_index = self.add_noises_around_points(new_points, new_weights, new_index)
        if self.add_random_weight_noise:
            new_points, new_weights, new_index = self.add_random_noise_to_weights(new_points, new_weights, new_index)
        if self.normalize_weights:
            new_weights = new_weights*(point_weights.sum()/(new_weights.sum()))
        if self.plot:
            visualize(points,new_points,point_weights,new_weights)

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
        self.do_local_deform_aug = aug_settings["do_local_deform_aug"]
        grid_aug_settings = self.aug_settings["grid_spline_aug"]
        local_deform_aug_settings = self.aug_settings["local_deform_aug"]
        grid_spline_kernel_obj = grid_aug_settings[("grid_spline_kernel_obj","","grid spline kernel object")]
        local_deform_spline_kernel_obj = local_deform_aug_settings[("local_deform_spline_kernel_obj","","local deform spline kernel object")]
        self.grid_spline_kernel = obj_factory(grid_spline_kernel_obj) if grid_spline_kernel_obj else None
        self.local_deform_spline_kernel = obj_factory(local_deform_spline_kernel_obj)  if local_deform_spline_kernel_obj else None
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

    def local_deform_spline_deform(self,points, point_weights):
        local_deform_aug_settings = self.aug_settings["local_deform_aug"]
        num_sample = local_deform_aug_settings["num_sample"]
        scale = local_deform_aug_settings["disp_scale"]
        sampler = uniform_sampler(num_sample,fixed_random_seed=False,sampled_by_weight=False)
        sampling_control_points,sampling_control_weights, _ = sampler(points, point_weights)
        sampling_control_disp = torch.ones_like(sampling_control_points).uniform_(-1,1)*scale
        points_disp = self.local_deform_spline_kernel(points[None], sampling_control_points[None], sampling_control_disp[None],sampling_control_weights[None])
        deformed_points =points + points_disp[0]
        return deformed_points, point_weights





    def __call__(self,batch_points, batch_point_weights):
        """
        :param points: torch.tensor  NxD
        :param point_weights: torch.tensor Nx1
        :return:
        """
        deformed_points_list, deformed_weights_list = [], []
        for points, point_weights in zip(batch_points, batch_point_weights):
            deformed_points = points
            deformed_weights = point_weights
            if self.do_local_deform_aug:
                deformed_points, deformed_weights = self.local_deform_spline_deform(deformed_points,deformed_weights)
            if self.plot:
                visualize(points,deformed_points,point_weights,deformed_weights)
            if self.do_grid_aug:
                deformed_points, deformed_weights = self.grid_spline_deform(deformed_points,deformed_weights)
            if self.plot:
                visualize(points,deformed_points,point_weights,deformed_weights)
            deformed_points_list.append(deformed_points)
            deformed_weights_list.append(deformed_weights)
        return torch.stack(deformed_points_list), torch.stack(deformed_weights_list)
