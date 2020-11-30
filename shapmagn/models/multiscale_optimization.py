from shapmagn.utils.point_sampler import grid_sampler
from shapmagn.utils.point_interpolator import kernel_interpolator
from shapmagn.modules.optimizer import optimizer_builder
from shapmagn.modules.scheduler import scheduler_builder

def build_multi_scale_solver(opt, model):
    """
    :param opt:
    :param model:
    :param optimizer:
    :return:
    """
    scale_list = opt[("scales", [0.1, 0.02], "a list of scales that parameterizes the voxel-grid sampling,"
                                             " the scale is from rough to fine resolution")]
    scale_iter_list = opt[("iter_per_scale", [100, 100], "number of iterations per scale")]
    scale_rel_ftol_list = opt[("rel_ftol_per_scale", [1e-4, 1e-4], "number of iterations per scale")]
    scale_sampler_list = [grid_sampler(scale) for scale in scale_list]
    num_scale = len(scale_list)
    interp_kernel_width_list = opt[
        ("interp_kernel_width_list", [0.1], "a list of kernel width that used to do upsampling")]
    scale_interpolator_list = [kernel_interpolator(interp_kernel_width)
                               for interp_kernel_width in interp_kernel_width_list]
    single_scale_solver_list = [build_single_scale_solver(opt,model, scale_iter_list[i], scale_rel_ftol_list[i])
                                for i in range(len(scale_list))]

    def solve(input_data):
        scale_output = None
        scale_input_data = scale_sampler_list[0](input_data, scale_list[0]) if scale_list[0] > 0 else input_data
        for i, en_scale in enumerate(scale_list):
            scale_output = single_scale_solver_list[i](scale_input_data)
            if i != num_scale - 1:
                scale_input_data = scale_sampler_list[i + 1](input_data, scale_list[i]) if scale_list[i+1]>0 else input_data
                input_data = scale_interpolator_list[i](scale_output, scale_input_data)
        return scale_output

    return solve


def build_single_scale_solver(opt,model, num_iter, rel_ftol=1e-4):
    def solve(input_data):
        opt_optim = opt['optim']
        """settings for the optimizer"""
        opt_scheduler = opt['scheduler']
        """settings for the scheduler"""
        optimizer = optimizer_builder(opt_optim)([input_data["reg_param"]])
        lr_scheduler = scheduler_builder(opt_scheduler)(optimizer)
        """initialize the optimizer and scheduler"""
        last_energy = 0.0
        for iter in range(num_iter):
            model.set_cur_iter(iter)
            output = model(input_data)
            cur_energy = model.cal_loss(output, input_data)
            cur_energy.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step(iter)
            cur_energy = cur_energy.item()
            rel_f = abs(last_energy - cur_energy) / (1 + abs(cur_energy))
            last_energy = cur_energy
            if rel_f < rel_ftol:
                print('Reached relative function tolerance of = ' + str(rel_ftol))
                break
        return model.get_output()

    return solve
