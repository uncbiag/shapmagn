import os
from shapmagn.modules.optimizer import optimizer_builder
from shapmagn.modules.scheduler import scheduler_builder
from shapmagn.global_variable import SHAPE_SAMPLER_POOL
from shapmagn.shape.shape_pair_utils import create_shape_pair,updater_for_shape_pair_from_low_scale
from shapmagn.utils.obj_factory import obj_factory
from shapmagn.utils.visual_utils import save_shape_pair_into_vtks


def build_multi_scale_solver(opt, model):
    """
    :param opt:
    :param model:
    :param optimizer:
    :return:
    """
    sampler_scale_list = opt[("scales", [0.1, -1], "a list of scales that parameterizes the voxel-grid sampling,"
                                             " the scale is from rough to fine resolution, -1 refers to the original resolution")]
    sampler_npoints_list = opt[("npoints", [1000, -1], "a list of scales that parameterizes the uniform sampling,"
                                             " the scale is from rough to fine resolution, -1 refers to the original resolution")]
    scale_iter_list = opt[("iter_per_scale", [100, 100], "number of iterations per scale")]
    scale_rel_ftol_list = opt[("rel_ftol_per_scale", [1e-4, 1e-4], "number of iterations per scale")]
    shape_sampler_type = opt[("shape_sampler_type", "point_grid", "shape sampler type: 'point_grid' or 'uniform'")]
    assert shape_sampler_type in ["point_grid","point_uniform"], "currently only point sampling {} is supported".format(["point_grid","point_uniform"])
    scale_args_list = sampler_scale_list if shape_sampler_type=='point_grid' else sampler_npoints_list
    scale_shape_sampler_list = [SHAPE_SAMPLER_POOL[shape_sampler_type](scale) for scale in scale_args_list]
    num_scale = len(scale_iter_list)
    updater_list = [updater_for_shape_pair_from_low_scale(model=model) for _ in range(num_scale-1)]
    source_target_generator = opt[("source_target_generator", "shape_pair_util.create_source_and_target_shape()","generator func")]
    source_target_generator = obj_factory(source_target_generator)
    single_scale_solver_list = [build_single_scale_solver(opt,model, scale_iter_list[i],scale_args_list[i], scale_rel_ftol_list[i])
                                for i in range(num_scale)]
    print("Multi-scale solver initialized!")
    print("The optimization works on the strategy '{}' with setting {}".format(shape_sampler_type, scale_args_list))

    def solve(input_data):
        source, target = source_target_generator(input_data)
        output_shape_pair = None
        for i, en_scale in enumerate(scale_args_list):
            print("{} th scale optimization begins, with  the strategy '{}' with setting {}".format(i, shape_sampler_type, scale_args_list[i]))
            scale_source = scale_shape_sampler_list[i](source) if scale_args_list[i] > 0 else source
            scale_target = scale_shape_sampler_list[i](target) if scale_args_list[i] > 0 else target
            toinput_shape_pair = create_shape_pair(scale_source, scale_target)
            if i != 0:
                toinput_shape_pair = updater_list[i-1](output_shape_pair, toinput_shape_pair)
                del output_shape_pair
            output_shape_pair = single_scale_solver_list[i](toinput_shape_pair)
        return output_shape_pair

    return solve


def build_single_scale_solver(opt,model, num_iter,scale=1, rel_ftol=1e-4, patient=5):
    def solve(shape_pair):
        save_every_n_iter = opt[("save_every_n_iter", 20, "save output every n iteration")]
        record_path = opt[("record_path", "./", "record path")]
        record_path = os.path.join(record_path,"scale_{}".format(scale))
        os.makedirs(record_path,exist_ok=True)
        opt_optim = opt['optim']
        """settings for the optimizer"""
        opt_scheduler = opt['scheduler']
        """settings for the scheduler"""
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
                patient_count = patient_count+1 if (iter-previous_converged_iter)==1 else 0
                previous_converged_iter = iter
                if patient_count>patient:
                    print('Reached relative function tolerance of = ' + str(rel_ftol))
                    break
        save_shape_pair_into_vtks(record_path, "iter_last", shape_pair)
        model.reset_iter()
        return shape_pair

    return solve

###################

def grad_hook(grad):
    # import pydevd
    # pydevd.settrace(suspend=False, trace_only_current_thread=True)
    print("the grad_norm is {} ".format(grad.norm()))
    return grad
############################3