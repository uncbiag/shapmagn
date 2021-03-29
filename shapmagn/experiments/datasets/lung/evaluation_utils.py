import torch
from shapmagn.global_variable import Shape

"""

copd10/12829U_EXP_STD_USD_COPD.nrrd
copd10/12829U_INSP_STD_USD_COPD.nrrd
copd1/13216S_EXP_STD_USD_COPD.nrrd
copd1/13216S_INSP_STD_USD_COPD.nrrd
copd2/13528L_EXP_STD_USD_COPD.nrrd
copd2/13528L_INSP_STD_USD_COPD.nrrd
copd3/13671Q_EXP_STD_USD_COPD.nrrd
copd3/13671Q_INSP_STD_USD_COPD.nrrd
copd4/13998W_EXP_STD_USD_COPD.nrrd
copd4/13998W_INSP_STD_USD_COPD.nrrd
copd5/17441T_EXP_STD_USD_COPD.nrrd
copd5/17441T_INSP_STD_USD_COPD.nrrd
copd6/12042G_EXP_STD_USD_COPD.nrrd
copd6/12042G_INSP_STD_USD_COPD.nrrd
copd7/12105E_EXP_STD_USD_COPD.nrrd
copd7/12105E_INSP_STD_USD_COPD.nrrd
copd8/12109M_EXP_STD_USD_COPD.nrrd
copd8/12109M_INSP_STD_USD_COPD.nrrd
copd9/12239Z_EXP_STD_USD_COPD.nrrd
copd9/12239Z_INSP_STD_USD_COPD.nrrd
"""



COPD={
"12042G":"copd6",
"12105E":"copd7",
"12109M":"copd8",
"12239Z":"copd9",
"12829U":"copd10",
"13216S":"copd1",
"13528L":"copd2",
"13671Q":"copd3",
"13998W":"copd4",
"17441T":"copd5"
}




def get_flowed(to_flowed_points, shape_pair, flow_fn):
    toflow = Shape()
    toflow.set_data(points=to_flowed_points)
    shape_pair.set_toflow(toflow)
    shape_pair.control_weights = torch.ones_like(shape_pair.control_weights) / shape_pair.control_weights.shape[1]
    flowed  = flow_fn(shape_pair)
    return flowed



def get_landmarks():
    pass



def evaluate_res():
    def eval(metrics, shape_pair, batch_info, additional_param=None, alias=''):
        source_name = batch_info["source_info"]["name"]
        target_name = batch_info["target_info"]["name"]

        sp, tp, fp = shape_pair.source.points, shape_pair.target.points, shape_pair.flowed.points
        if additional_param is not None and "mapped_position" in additional_param:
            fp = additional_param["mapped_position"]
    return




