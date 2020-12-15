import os
from shapmagn.modules.optimizer import optimizer_builder
from shapmagn.modules.scheduler import scheduler_builder
from shapmagn.global_variable import SHAPE_SAMPLER_POOL
from shapmagn.shape.shape_pair_utils import create_shape_pair,updater_for_shape_pair_from_low_scale
from shapmagn.utils.visual_utils import save_shape_pair_into_vtks


def build_multi_scale_solver(opt, model):
    """
    :param opt:
    :param model:
    :param optimizer:
    :return:
    """
    sampler_scale_list = opt[("point_grid_scales", [0.1, -1], "a list of scales that parameterizes the voxel-grid sampling,"
                                             " the scale is from rough to fine resolution, -1 refers to the original resolution")]
    sampler_npoints_list = opt[("point_uniform_npoints", [1000, -1], "a list of scales that parameterizes the uniform sampling,"
                                             " the scale is from rough to fine resolution, -1 refers to the original resolution")]
    scale_iter_list = opt[("iter_per_scale", [100, 100], "number of iterations per scale")]
    scale_iter_list = scale_iter_list if not model.call_thirdparty_package else [1 for _ in scale_iter_list]
    scale_rel_ftol_list = opt[("rel_ftol_per_scale", [1e-4, 1e-4], "number of iterations per scale")]
    shape_sampler_type = opt[("shape_sampler_type", "point_grid", "shape sampler type: 'point_grid' or 'uniform'")]
    assert shape_sampler_type in ["point_grid","point_uniform"], "currently only point sampling {} is supported".format(["point_grid","point_uniform"])
    scale_args_list = sampler_scale_list if shape_sampler_type=='point_grid' else sampler_npoints_list
    scale_shape_sampler_list = [SHAPE_SAMPLER_POOL[shape_sampler_type](scale) for scale in scale_args_list]
    num_scale = len(scale_iter_list)
    stragtegy = opt[("stragtegy", "use_solver_from_model','use_solver_defined_here")]
    build_single_scale_solver = build_single_scale_custom_solver if stragtegy =="use_solver_defined_here" else build_single_scale_model_embedded_solver()
    reg_param_initializer = model.init_reg_param
    param_updater = model.update_reg_param_from_low_scale_to_high_scale
    infer_shape_pair_after_upsampling = model.flow
    single_scale_solver_list = [build_single_scale_solver(opt,model, scale_iter_list[i],scale_args_list[i], scale_rel_ftol_list[i])
                                for i in range(num_scale)]
    print("Multi-scale solver initialized!")
    print("The optimization works on the strategy '{}' with setting {}".format(shape_sampler_type, scale_args_list))

    def solve(toinput_shape_pair):
        source, target = toinput_shape_pair.source, toinput_shape_pair.target
        output_shape_pair = None
        for i in range(num_scale):
            print("{} th scale optimization begins, with  the strategy '{}' with setting {}".format(i, shape_sampler_type, scale_args_list[i]))
            scale_source = scale_shape_sampler_list[i](source) if scale_args_list[i] > 0 else source
            scale_target = scale_shape_sampler_list[i](target) if scale_args_list[i] > 0 else target
            toinput_shape_pair = create_shape_pair(scale_source, scale_target)
            reg_param_initializer(toinput_shape_pair)
            if i != 0:
                toinput_shape_pair = param_updater(output_shape_pair, toinput_shape_pair)
                del output_shape_pair
            output_shape_pair = single_scale_solver_list[i](toinput_shape_pair)
        if scale_args_list[-1]!=-1:
            output_shape_pair = param_updater(output_shape_pair, create_shape_pair(source, target))
            output_shape_pair = infer_shape_pair_after_upsampling(output_shape_pair)
        return output_shape_pair

    return solve


def build_single_scale_custom_solver(opt,model, num_iter,scale=-1, rel_ftol=1e-4, patient=5):
    """
    custom solver where the param needs to optimize iteratively
    this is typically required by native displacement method, lddmm method,

    :param opt:
    :param model:
    :param num_iter:
    :param scale:
    :param rel_ftol:
    :param patient:
    :return:
    """
    save_every_n_iter = opt[("save_every_n_iter", 20, "save output every n iteration")]
    record_path = opt[("record_path", "", "record path")]
    record_path = os.path.join(record_path, "scale_{}".format(scale))
    os.makedirs(record_path, exist_ok=True)
    opt_optim = opt['optim']
    """settings for the optimizer"""
    opt_scheduler = opt['scheduler']
    """settings for the scheduler"""
    def solve(shape_pair):
        ######################################3
        shape_pair.reg_param.register_hook(grad_hook)
        ############################################3
        optimizer = optimizer_builder(opt_optim)([shape_pair.reg_param])
        lr_scheduler = scheduler_builder(opt_scheduler)(optimizer)
        """initialize the optimizer and scheduler"""
        last_energy = 0.0
        patient_count = 0
        previous_converged_iter =0.
        def closure():
            optimizer.zero_grad()
            cur_energy = model(shape_pair)
            cur_energy.backward()
            return cur_energy

        for iter in range(num_iter):
            cur_energy = optimizer.step(closure)
            lr_scheduler.step(iter)
            cur_energy = cur_energy.item()
            rel_f = abs(last_energy - cur_energy) / (abs(cur_energy))
            last_energy = cur_energy
            if iter%save_every_n_iter==0:
                save_shape_pair_into_vtks(record_path, "iter_{}".format(iter), shape_pair)
            if rel_f < rel_ftol:
                print("the converge rate: {} is too small".format(rel_f))
                patient_count = patient_count+1 if (iter-previous_converged_iter)==1 else 0
                previous_converged_iter = iter
                if patient_count>patient:
                    print('Reached relative function tolerance of = ' + str(rel_ftol))
                    break
        save_shape_pair_into_vtks(record_path, "iter_last", shape_pair)
        model.reset()
        return shape_pair

    return solve


def build_single_scale_model_embedded_solver(opt,model, num_iter=1,scale=-1, rel_ftol=1e-4, patient=2):
    """
    the optimizer and scheduler are not defined
    the model take responsibility to solve the optimal solution
    this is typically required by thirdparty package, least square regression, gradient flow,
     or approaches needs inner iteration ,e.g, coherent point drift
    :param opt:
    :param model:
    :param num_iter: typically the iteration here should be set to 1, but here we leave it for potential usage
    :param scale:
    :param rel_ftol:
    :param patient:
    :return:
    """
    save_every_n_iter = opt[("save_every_n_iter", 1, "save output every n iteration")]
    record_path = opt[("record_path", "", "record path")]
    record_path = os.path.join(record_path, "scale_{}".format(scale))
    os.makedirs(record_path, exist_ok=True)
    def solve(shape_pair):
        last_energy = 0.0
        patient_count = 0
        previous_converged_iter = 0.
        for iter in range(num_iter):
            cur_energy = model(shape_pair)
            cur_energy = cur_energy.item()
            rel_f = abs(last_energy - cur_energy) / (abs(cur_energy))
            last_energy = cur_energy
            if iter % save_every_n_iter == 0:
                save_shape_pair_into_vtks(record_path, "iter_{}".format(iter), shape_pair)
            if rel_f < rel_ftol:
                print("the converge rate: {} is too small".format(rel_f))
                patient_count = patient_count + 1 if (iter - previous_converged_iter) == 1 else 0
                previous_converged_iter = iter
                if patient_count > patient:
                    print('Reached relative function tolerance of = ' + str(rel_ftol))
                    break
        save_shape_pair_into_vtks(record_path, "iter_last", shape_pair)
        model.reset()
        return shape_pair
    return solve

###################

def grad_hook(grad):
    # import pydevd
    # pydevd.settrace(suspend=False, trace_only_current_thread=True)
    print("the grad_norm is {} ".format(grad.norm()))
    return grad
############################3