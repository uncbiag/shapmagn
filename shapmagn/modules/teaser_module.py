"""
Teaser is disabled, it only works when the point correspondence is given
which doesn't meet the general settings of shapmagn
"""

import numpy as np
try:
    import teaserpp_python
except:
    print("Teaser is not detected, related functions are disabled")
import torch
class Teaser(object):
    """
    Disabled
    Teaser only works when the point correspondence is given,
    todo to make teaser work, additional feature extraction step is needed.
    scaled rigid registration


    default settting in json
    "teaser": {
                "cbar2": 1,
                "noise_bound": 0.1,
                "estimate_scaling": true,
                "rotation_gnc_factor":1.4,
                "rotation_max_iterations": 100,
                "rotation_cost_threshold": 1e-12
            }
    """
    def __init__(self, opt):
        super(Teaser,self).__init__()
        self.opt = opt
        cbar2 = opt[("cbar2", 1,"teaser params")]
        noise_bound = opt[("noise_bound", 0.01,"teaser params")]
        estimate_scaling = opt[("estimate_scaling", True,"teaser params")]
        rotation_gnc_factor = opt[("rotation_gnc_factor", 1.4,"teaser params")]
        rotation_max_iterations = opt[("rotation_max_iterations", 100,"teaser params")]
        rotation_cost_threshold = opt[("rotation_cost_threshold", 1e-12,"teaser params")]
        solver_params = teaserpp_python.RobustRegistrationSolver.Params()
        solver_params.cbar2 = cbar2
        solver_params.noise_bound = noise_bound
        solver_params.estimate_scaling = estimate_scaling
        solver_params.rotation_estimation_algorithm = (
            teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        )
        solver_params.rotation_gnc_factor = rotation_gnc_factor
        solver_params.rotation_max_iterations = rotation_max_iterations
        solver_params.rotation_cost_threshold = rotation_cost_threshold
        print("TEASER++ Parameters are:", solver_params)
        self.solver = teaserpp_python.RobustRegistrationSolver(solver_params)
        self.prealign=True



    def set_mode(self, mode=None):
        pass

    def _get_input(self, source_batch, target_batch):
        source_list = [source.transpose() for source in source_batch.detach().cpu().numpy()]
        target_list = [target.transpose() for target in target_batch.detach().cpu().numpy()]
        return source_list, target_list


    def _convert_transform_format(self,solution_list, device):
        transform_matrix_list = [(solution.rotation*solution.scale).transpose for solution in solution_list]
        translation_list = [(solution.translation)[None] for solution in solution_list]
        batch_transform_matrix = torch.from_numpy(np.stack(transform_matrix_list))
        batch_translation = torch.from_numpy(np.stack(translation_list))
        return_param = torch.cat((batch_transform_matrix,batch_translation),dim=1)
        return return_param.to(device)


    def __call__(self,source_batch, target_batch):
        """

        :param source_batch: BxNxD
        :param target_batch: BxNxD
        :return: BxDx(D+1) transform matrix
        """
        device = source_batch.device
        source_list, target_list = self._get_input(source_batch, target_batch)
        solution_list = []
        for source, target in zip(source_list, target_list):
            self.solver.solve(source, target)
            solution = self.solver.getSolution()
            solution_list.append(solution)
        return self._convert_transform_format(solution_list,device)