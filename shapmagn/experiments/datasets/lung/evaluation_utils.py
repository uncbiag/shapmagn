import os
import numpy as np
import torch
from shapmagn.global_variable import Shape
from shapmagn.datasets.vtk_utils import read_vtk
from shapmagn.shape.point_interpolator import NadWatIsoSpline
from shapmagn.utils.shape_visual_utils import save_shape_into_files

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



ID_COPD={
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

CENTER={
"13216S_INSP_STD_USD_COPD":[   7.979657,   25.017563, -151.31465 ],
"13216S_EXP_STD_USD_COPD":[   8.846239,   45.1596,   -142.66893 ],
"17441T_INSP_STD_USD_COPD":[  13.640025,   -9.945671, -186.71013 ],
"17441T_EXP_STD_USD_COPD":[  12.206295,    8.053513, -165.32997 ],
"12109M_INSP_STD_USD_COPD":[   7.076656,     6.5697513, -167.6756   ],
"12109M_EXP_STD_USD_COPD":[   7.3253126,   13.625545,  -146.99274  ],
"13998W_INSP_STD_USD_COPD":[  25.451248,     1.2760051, -136.3838   ],
"13998W_EXP_STD_USD_COPD":[  22.506023,   17.911581, -109.80095 ],
"12042G_INSP_STD_USD_COPD":[-7.9421997e-03, -2.8869128e+00, -1.4221332e+02],
"12042G_EXP_STD_USD_COPD":[   0.782543,   12.822629, -130.49344 ],
"12239Z_INSP_STD_USD_COPD":[   9.527761,    4.727795, -148.14838 ],
"12239Z_EXP_STD_USD_COPD":[  13.590356,    9.209801, -135.56178 ],
"13528L_INSP_STD_USD_COPD":[ -11.987083,   13.766904, -119.20886 ],
"13528L_EXP_STD_USD_COPD":[ -13.89523,    23.859629, -122.09784 ],
"12105E_INSP_STD_USD_COPD":[   8.279412,    5.61014,  -161.163   ],
"12105E_EXP_STD_USD_COPD":[  10.5092535,   10.868305,  -150.65265  ],
"13671Q_INSP_STD_USD_COPD":[  13.88625,      7.0715256, -174.34314  ],
"13671Q_EXP_STD_USD_COPD":[  15.094385,    10.8874855, -162.57578  ],
"12829U_INSP_STD_USD_COPD":[   1.1542492,   11.651825,  -163.67746  ],
"12829U_EXP_STD_USD_COPD":[   5.068997,   15.700953, -145.50748 ]

}

SCALE=100



dirlab_landmarks_folder_path  = "/playpen-raid1/Data/copd/processed/landmark_processed"
def get_flowed(shape_pair, interp_kernel):
    flowed_points = interp_kernel(shape_pair.toflow.points, shape_pair.source.points, shape_pair.flowed.points, shape_pair.source.weights)
    flowed = Shape()
    flowed.set_data_with_refer_to(flowed_points, shape_pair.toflow)
    shape_pair.set_flowed(flowed)
    return shape_pair



def get_landmarks(source_landmarks_path,target_landmarks_path):
    source_landmarks = read_vtk(source_landmarks_path)["points"]
    target_landmarks = read_vtk(target_landmarks_path)["points"]
    return source_landmarks, target_landmarks





def eval_landmark(model,shape_pair, batch_info,alias, eval_ot_map=False):
    s_name_list = batch_info["source_info"]["name"]
    t_name_list = batch_info["target_info"]["name"]
    landmarks_toflow_list, target_landmarks_list = [], []
    for s_name, t_name in zip(s_name_list, t_name_list):
        source_landmarks_path = os.path.join(dirlab_landmarks_folder_path,s_name+".vtk")
        target_landmarks_path = os.path.join(dirlab_landmarks_folder_path,t_name+".vtk")
        landmarks_toflow, target_landmarks = get_landmarks(source_landmarks_path,target_landmarks_path)
        landmarks_toflow = (landmarks_toflow-np.array(CENTER[s_name]))/SCALE
        target_landmarks = (target_landmarks-np.array(CENTER[t_name]))/SCALE
        landmarks_toflow_list.append(landmarks_toflow)
        target_landmarks_list.append(target_landmarks)
    device =  shape_pair.source.points.device
    flowed_cp = shape_pair.flowed
    landmarks_toflow = torch.Tensor(np.stack(landmarks_toflow_list,0)).to(device)
    target_landmarks_points = torch.Tensor(np.stack(target_landmarks_list,0)).to(device)
    shape_pair.toflow = Shape().set_data(points=landmarks_toflow, weights= torch.ones_like(landmarks_toflow))
    gt_landmark = Shape().set_data(points=target_landmarks_points, weights= torch.ones_like(target_landmarks_points))
    if not eval_ot_map:
        shape_pair = model.flow(shape_pair)
    else:
        flowed_points = NadWatIsoSpline(exp_order=2,kernel_scale=0.005)
        shape_pair = get_flowed(shape_pair,flowed_points)
    flowed_landmarks_points = shape_pair.flowed.points
    record_path = os.path.join(batch_info["record_path"], "3d", "{}_epoch_{}".format(batch_info["phase"],batch_info["epoch"]))
    save_shape_into_files(record_path, "landmark"+alias+"_toflow",batch_info["pair_name"], shape_pair.toflow)
    save_shape_into_files(record_path, "landmark"+alias+"_flowed",batch_info["pair_name"], shape_pair.flowed)
    save_shape_into_files(record_path, "landmark"+alias+"_target",batch_info["pair_name"], gt_landmark)
    shape_pair.flowed  = flowed_cp # compatible to save function
    shape_pair.toflow = None  # compatible to save function
    return (target_landmarks_points - flowed_landmarks_points)*SCALE




def evaluate_res():
    def eval(metrics, shape_pair, batch_info, additional_param=None, alias=''):
        phase = batch_info["phase"]
        if phase=="val" or phase=="test":
            model = additional_param["model"]
            flowed_points_cp= shape_pair.flowed.points
            shape_pair.control_points = additional_param["initial_control_points"]
            eval_ot_map = "mapped_position" in additional_param
            if additional_param is not None and eval_ot_map:
                shape_pair.flowed.points = additional_param["mapped_position"]
            diff = eval_landmark(model, shape_pair,batch_info,alias, eval_ot_map=eval_ot_map)
            diff_var = (diff-diff.mean(1,keepdim=True))**2
            diff_var = diff_var.sum(2).mean(1)
            diff_norm_mean = diff.norm(p=2,dim=2).mean(1)
            metrics.update({"lmk_diff_mean"+alias:[_diff_norm_mean.item() for _diff_norm_mean in diff_norm_mean],
                            "lmk_diff_var"+alias:[_diff_var.item() for _diff_var in diff_var]})
            shape_pair.flowed.points = flowed_points_cp
        return metrics
    return eval



