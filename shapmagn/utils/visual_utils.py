import os
import torch
import pyvista as pv



def save_points_into_vtk(folder_path, name, points, weights=None):
    if isinstance(points, torch.Tensor):
        points = points.detach().numpy()
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().numpy()
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
