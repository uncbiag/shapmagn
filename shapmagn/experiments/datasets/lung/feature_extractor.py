import numpy as np
import torch
from pykeops.torch import LazyTensor
from pykeops.torch.cluster import grid_cluster, sort_clusters, cluster_ranges_centroids, from_matrix
import pyvista as pv
from shapmagn.utils.visualizer import visualize_point_fea

def numpy(x):
    return x.detach().cpu().numpy()

def compute_local_moments(points, radius=1, ranges=None):
    # B, N, D = points.shape
    shape_head, D = points.shape[:-1], points.shape[-1]

    scale = 1.41421356237 * radius  # math.sqrt(2) is not super JIT-friendly...
    x = points / scale  # Normalize the kernel size

    # Computation:
    x = torch.cat((torch.ones_like(x[...,:1]), x), dim = -1)  # (B, N, D+1)

    x_i = LazyTensor(x[...,:,None,:])  # (B, N, 1, D+1)
    x_j = LazyTensor(x[...,None,:,:])  # (B, 1, N, D+1)

    D_ij = ((x_i - x_j) ** 2).sum(-1)  # (B, N, N), squared distances
    K_ij = (- D_ij).exp()  # (B, N, N), Gaussian kernel

    C_ij = (K_ij * x_j).tensorprod(x_j)  # (B, N, N, (D+1)*(D+1))

    C_ij.ranges = ranges
    C_i  = C_ij.sum(dim = len(shape_head)).view(shape_head + (D+1, D+1))  # (B, N, D+1, D+1) : descriptors of order 0, 1 and 2

    w_i = C_i[...,:1,:1]               # (B, N, 1, 1), weights
    m_i = C_i[...,:1,1:] * scale       # (B, N, 1, D), sum
    c_i = C_i[...,1:,1:] * (scale**2)  # (B, N, D, D), outer products

    mass_i = w_i.squeeze(-1).squeeze(-1)  # (B, N)
    dev_i = (m_i / w_i).squeeze(-2) - points  # (B, N, D)
    cov_i  = (c_i - (m_i.transpose(-1, -2) * m_i) / w_i) / w_i  # (B, N, D, D)

    return mass_i, dev_i, cov_i


def compute_local_fea(fea_type_list, raidus=1, normalize=True):
    eps = 0
    def _compute_local_fea(fea_type, mass,dev,cov):
        if fea_type=="mass":
            fea = mass
        elif fea_type == "dev":
            fea = (dev ** 2).sum(-1)
        elif fea_type == "eignvalue":
            vals = cov.symeig(eigenvectors=False).eigenvalues
            fea = (mass ** 2) * vals[:, 1] * vals[:, 2]
        elif fea_type=="eignvector":
            vector = cov.symeig(eigenvectors=True).eigenvectors
            fea = vector[:,:,0]

        if normalize:
            fea = _normalize_fea(fea)
        return fea
    def _normalize_fea(fea):
        fea = fea.clamp(0, 10).sqrt() #* (mass > 2)
        return fea

    def _compute(points):
        if isinstance(points, np.ndarray):
            points =torch.from_numpy(points)
        xyz_labels = grid_cluster(points, eps)  # class labels
        xyz_ranges, xyz_centroids, _ = cluster_ranges_centroids(points, xyz_labels)
        xyz, xyz_labels = sort_clusters(points, xyz_labels)
        D = ((xyz_centroids[..., :, None, :] - xyz_centroids[..., None, :, :]) ** 2).sum(-1)
        keep = D < (4 * raidus + 2 * eps) ** 2
        ranges_xyz = from_matrix(xyz_ranges, xyz_ranges, keep)
        mass, dev, cov = compute_local_moments(xyz, radius=raidus, ranges=ranges_xyz)  # (N,), (N, D), (N, D, D)
        fea_combined = [_compute_local_fea(fea_type, mass, dev, cov) for fea_type in fea_type_list]
        fea_combined = torch.stack(fea_combined,1)
        return fea_combined.numpy()
    return _compute














if __name__ == "__main__":
    cloud_1 = pv.read(
        "/playpen-raid1/Data/UNC_vesselParticles/10005Q_EXP_STD_NJC_COPD_wholeLungVesselParticles.vtk")
    cloud_2 = pv.read(
        "/playpen-raid1/Data/UNC_vesselParticles/10005Q_INSP_STD_NJC_COPD_wholeLungVesselParticles.vtk")
    saving_path = "/playpen-raid1/zyshen/debug/debugg_point_visual2.vtk"
    points_1 = cloud_1.points
    points_2 = cloud_2.points
    # fea_type_list = ["mass","dev","eignvalue"]
    fea_type_list = ["eignvector"]
    compute_fea = compute_local_fea(fea_type_list,raidus=50,normalize=False)
    combined_fea1= compute_fea(points_1,)
    combined_fea2= compute_fea(points_2)
    visual_scalars1 = combined_fea1[:,0]
    visual_scalars1 = (visual_scalars1 - visual_scalars1.min(0)) / (visual_scalars1.max(0) - visual_scalars1.min(0))
    fea_index = 0
    # visual_scalars1 = combined_fea1[:,fea_index]
    # visual_scalars2 = combined_fea2[:,fea_index]
    #visual_scalars1 = (points_1-points_1.min(0))/(points_1.max(0)-points_1.min(0))
    visualize_point_fea(points_1, visual_scalars1)
    #visualize_point_fea(points_2,visual_scalars2)

