# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import torch
import os
import numpy as np
import trimesh
import pyvista as pv
from shapmagn.shape.shape_utils import get_scale_and_center

PROGRAM_PATH="./"
get_shape_path = lambda shape_name: os.path.join(os.path.join(PROGRAM_PATH,"simple_shape",shape_name,"train",shape_name))+'.off'

def read_off(fpath):
    shape = trimesh.load(fpath)
    return  shape.vertices, shape.faces


def get_shape(shape_name):
    shape_path = get_shape_path(shape_name)
    verts, faces = read_off(shape_path)
    return verts, faces




def normalize_vertice(vertices):
    scale, shift = get_scale_and_center(vertices,percentile=100)
    vertices = (vertices-shift)/scale
    return  vertices

def subdivide(vertices, faces, level=2):
    for _ in range(level):
        vertices, faces = trimesh.remesh.subdivide(vertices, faces)
    return vertices, faces

def transfer_faces_into_vtk_format(faces):
    ind = np.ones([faces.shape[0],1])*3
    faces = np.concatenate((ind,faces),1).astype(np.int64)
    return faces.flatten()

def compute_interval(vertices):
    vert_i  = vertices[:,None]
    vert_j  = vertices[None]
    vert_dist = ((vert_i-vert_j)**2).sum(-1)
    vert_dist = np.sqrt(vert_dist)
    print("the min interval is {}".format(np.min(vert_dist[np.where(vert_dist>0)])))




if __name__ =="__main__":
    shape_name = "3d_sphere"
    level =1
    saving_path = "/playpen-raid1/zyshen/debug/shapmagn/divide_{}_level{}.vtk".format(shape_name,level)
    verts, faces = get_shape(shape_name)
    verts, faces =subdivide(verts, faces, level=level)
    verts = normalize_vertice(verts)
    faces = transfer_faces_into_vtk_format(faces)
    compute_interval(verts)
    data = pv.PolyData(verts,faces)
    data.save(saving_path)
    shape_name = "3d_cube"
    level = 4
    saving_path = "/playpen-raid1/zyshen/debug/shapmagn/divide_{}_level{}.vtk".format(shape_name, level)
    verts, faces = get_shape(shape_name)
    verts, faces = subdivide(verts, faces, level=level)
    verts = normalize_vertice(verts)
    faces = transfer_faces_into_vtk_format(faces)
    compute_interval(verts)
    data = pv.PolyData(verts, faces)
    data.save(saving_path)


    """
    the min interval is 0.0191220190793802
    the min interval is 0.125
    """


