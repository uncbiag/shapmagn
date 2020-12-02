import os
import torch
import pyvista as pv



def save_points_into_vtk(folder_path, name, points, weights=None):
    if isinstance(points, torch.Tensor):
        points = points.cpu().detach().numpy()
    if isinstance(weights, torch.Tensor):
        weights = weights.cpu().detach().numpy()
    if len(points.shape)==3 and len(weights.shape)==3:
        nbatch,_,_ = points.shape
    elif len(points.shape)==2 and len(weights.shape)==2:
        points = points[None]
        weights = weights[None]
        nbatch = 1
    else:
        raise ValueError("shape not supported")
    os.makedirs(folder_path,exist_ok=True)
    for b in range(nbatch):
        data = pv.PolyData(points[b])
        if weights is not None:
            data.point_arrays['weights'] = weights[b]
        fpath = os.path.join(folder_path, name)+"_{}.vtk".format(b)
        data.save(fpath)


def save_shape_into_vtks(folder_path,name, shape):
    save_points_into_vtk(folder_path,name,shape.points,shape.weights)


# from shapmagn.utils.visual_utils import save_shape_pair_into_vtks
def save_shape_pair_into_vtks(folder_path, name, shape_pair):
    folder_path = os.path.join(folder_path,name)
    save_shape_into_vtks(folder_path,"source_weight",shape_pair.source)
    save_shape_into_vtks(folder_path,"target_weight",shape_pair.target)
    if shape_pair.flowed is not None:
        save_shape_into_vtks(folder_path, "flowed_weight", shape_pair.flowed)
        save_shape_into_vtks(folder_path, "toflow_weight", shape_pair.toflow)
    if shape_pair.control_points is not None:
        save_points_into_vtk(folder_path,"control_weight",shape_pair.control_points,shape_pair.control_weights)
    if shape_pair.flowed_control_points is not None:
        save_points_into_vtk(folder_path,"flowed_control_weight",shape_pair.flowed_control_points,shape_pair.control_weights)
    if shape_pair.reg_param is not None:
        reg_param_norm = shape_pair.reg_param.norm(p=2,dim=2,keepdim=True)
        save_points_into_vtk(folder_path,"control_reg_param_norm",shape_pair.control_points,reg_param_norm)
        save_points_into_vtk(folder_path,"control_reg_param_vector",shape_pair.control_points,shape_pair.reg_param)

