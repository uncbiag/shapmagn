import os
import torch
import pyvista as pv
from shapmagn.datasets.vtk_utils import convert_faces_into_vtk_format



def save_shape_into_vtk(folder_path, name, **args):
    for key, item in args.items():
        if isinstance(item, torch.Tensor):
            args[key] = item.cpu().detach().numpy()
    points = args["points"]
    faces = args["faces"] if "faces" in args else None
    if len(points.shape)==3:
        nbatch,_,_ = points.shape
    elif len(points.shape)==2:
        for key, item in args.items():
                args[key] = args[key][None]
        nbatch = 1
    else:
        raise ValueError("shape not supported")
    os.makedirs(folder_path,exist_ok=True)
    for b in range(nbatch):
        if faces is not None:
            face = convert_faces_into_vtk_format(faces[b])
            data = pv.PolyData(points[b],face)
        else:
            data = pv.PolyData(points[b])
        for key, item in args.items():
            if key not in ['points','faces']:
                data.point_arrays[key] = item[b]
        fpath = os.path.join(folder_path, name)+"_{}.vtk".format(b)
        data.save(fpath)



def save_shape_into_vtks(folder_path,name, shape):

    attri_dict_to_save = {"points":shape.points, "weights":shape.weights}
    attri_dict_to_save["faces"] = shape.faces if not shape.points_mode_on else None
    if shape.pointfea is not None:
        attri_dict_to_save["pointfea"] = torch.norm(shape.pointfea, 2, dim=2)
    save_shape_into_vtk(folder_path,name, **attri_dict_to_save)


def save_shape_pair_into_vtks(folder_path, name, shape_pair):
    folder_path = os.path.join(folder_path,name)
    save_shape_into_vtks(folder_path,"source_weight",shape_pair.source)
    save_shape_into_vtks(folder_path,"target_weight",shape_pair.target)
    if shape_pair.flowed is not None:
        save_shape_into_vtks(folder_path, "flowed_weight", shape_pair.flowed)
        save_shape_into_vtks(folder_path, "toflow_weight", shape_pair.toflow)
    if shape_pair.control_points is not None:
        save_shape_into_vtk(folder_path,"control_weight",**{"points":shape_pair.control_points,"weights":shape_pair.control_weights})
    if shape_pair.flowed_control_points is not None:
        save_shape_into_vtk(folder_path,"flowed_control_weight",**{"points":shape_pair.flowed_control_points,"weights":shape_pair.control_weights})
    if shape_pair.reg_param is not None:
        reg_param_norm = shape_pair.reg_param.norm(p=2,dim=2,keepdim=True)
        save_shape_into_vtk(folder_path,"reg_param",**{"points":shape_pair.control_points,"reg_param_norm":reg_param_norm, "reg_param_vector":shape_pair.reg_param})

