from shapmagn.utils.sampler import grid_sampler, kernel_interpolator

def build_multi_scale_solver(opt, input_data):
        scale_list = opt[("scales",[0.1,0.02],"a list of scales that parameterizes the voxel-grid sampling,"
                                              " the scale is from rough to fine resolution")]
        scale_iteration_list = opt[("iter_per_scale",[100,100],"number of iterations per scale")]
        scale_sampler_list = [grid_sampler(scale) for scale in scale_list]
        num_scale = len(scale_list)
        interp_kernel_width_list = opt[("interp_kernel_width_list",[0.1],"a list of kernel width that used to do upsampling")]
        scale_interpolator_list = [kernel_interpolator(interp_kernel_width) for interp_kernel_width in interp_kernel_width_list]

        #
        # for i, en_scale in enumerate(scale_list):
        #     scale_input_data = get_scale_input_data(input_data)
        #     scale_optimizer = build_scale_optimizer()
        #     single_scale_solver =  build_single_scale_solver()
        #     if i != num_scale -1:
        #         upsample_reg_param
        #
        #
        #

