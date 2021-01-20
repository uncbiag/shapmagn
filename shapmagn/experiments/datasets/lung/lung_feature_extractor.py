import numpy as np
import torch
from pykeops.torch import LazyTensor
from pykeops.torch.cluster import grid_cluster, sort_clusters, cluster_ranges_centroids, from_matrix
import pyvista as pv
from shapmagn.utils.visualizer import visualize_point_fea
from torchvectorized.vlinalg import vSymEig

def numpy(x):
    return x.detach().cpu().numpy()

def compute_local_moments(points, radius=1.):
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
    C_i  = C_ij.sum(dim = len(shape_head)).view(shape_head + (D+1, D+1))  # (B, N, D+1, D+1) : descriptors of order 0, 1 and 2

    w_i = C_i[...,:1,:1]               # (B, N, 1, 1), weights
    m_i = C_i[...,:1,1:] * scale       # (B, N, 1, D), sum
    c_i = C_i[...,1:,1:] * (scale**2)  # (B, N, D, D), outer products

    mass_i = w_i.squeeze(-1)  # (B, N)
    dev_i = (m_i / w_i).squeeze(-2) - points  # (B, N, D)
    cov_i  = (c_i - (m_i.transpose(-1, -2) * m_i) / w_i) / w_i  # (B, N, D, D)

    return mass_i, dev_i, cov_i


def compute_local_fea(fea_type_list, raidus=1.):
    def _compute_local_fea(fea_type, mass,dev,cov):
        if fea_type=="mass":
            fea = mass
        elif fea_type == "dev":
            fea = (dev ** 2).sum(-1,keepdim=True)
        elif 'eign' in fea_type:
            #vals = cov.symeig(eigenvectors=False).eigenvalues
            #vectors = cov.symeig(eigenvectors=True).eigenvectors
            compute_eign_vector = fea_type in ["eignvector"]
            B, N = cov.shape[0], cov.shape[1]
            cov = cov.view(B, N, -1, 1, 1).permute(0, 2, 1, 3, 4)
            vals, vectors = vSymEig(cov, eigenvectors=compute_eign_vector, flatten_output=True)
            vals = vals.view(B, N, -1)
            if fea_type == "eignvalue_prod":
                fea = mass* vals[..., 1:2]* vals[..., 2:3]
            elif fea_type == "eignvalue_cat":
                #fea = torch.cat([vals[..., 1:2], vals[..., 2:3]],-1)
                fea = mass* vals
            elif fea_type=="eignvector":
                vectors = vectors.view(B, N, 3, 3)
                fea = vectors[...,0]
                fea = torch.sign(fea[:,:,2:3]) * fea
        else:
            raise ValueError("not implemented")
        return fea


    def _compute(points, return_stats=False, mean=None, std=None):
        if isinstance(points, np.ndarray):
            points =torch.from_numpy(points)
        mass, dev, cov = compute_local_moments(points, radius=raidus)  # (N,), (N, D), (N, D, D)
        fea_combined = [_compute_local_fea(fea_type, mass, dev, cov) for fea_type in fea_type_list]
        fea_combined = torch.cat(fea_combined,-1)
        if mean is None and std is None:
            mean = fea_combined.mean(1,keepdim=True)
            std = fea_combined.std(1,keepdim=True)
        fea_combined = (fea_combined-mean)/std
        fea_combined = fea_combined.clamp(-2,2)
        if return_stats:
            return fea_combined, mean, std, mass
        else:
            return fea_combined, mass
    return _compute







def feature_extractor(fea_type_list, radius=0.01):
    lung_feature_extractor = compute_local_fea(fea_type_list,radius)
    def extract(flowed, target):
        flowed.pointfea, mean, std, _ = lung_feature_extractor(flowed.points, return_stats=True)
        target.pointfea, _ = lung_feature_extractor(target.points, mean=mean, std= std)
        return flowed, target
    return extract








if __name__ == "__main__":
    cloud_1 = pv.read(
        "/home/zyshen/remote/llr11_mount/Data/UNC_vesselParticles/10005Q_EXP_STD_NJC_COPD_wholeLungVesselParticles.vtk")
    cloud_2 = pv.read(
        "/home/zyshen/remote/llr11_mount/Data/UNC_vesselParticles/10005Q_INSP_STD_NJC_COPD_wholeLungVesselParticles.vtk")
    saving_path = "/playpen-raid1//home/zyshen/remote/llr11_mount/zyshen/debug/debugg_point_visual2.vtk"
    points_1 = cloud_1.points[:10000][None]
    points_2 = cloud_2.points[None]
    # fea_type_list = ["mass","dev","eignvalue_cat","eignvalue_prod","eignvector]
    fea_type_list = ["eignvector"]
    compute_fea = compute_local_fea(fea_type_list,raidus=1)
    combined_fea1, mass1= compute_fea(points_1)
    combined_fea2, mass2= compute_fea(points_2)
    filter1 = mass1[...,0]>2.5
    points_1 = points_1[filter1]
    combined_fea1 = combined_fea1[filter1]

    #visual_scalars1 = (visual_scalars1 - visual_scalars1.min(0)) / (visual_scalars1.max(0) - visual_scalars1.min(0))
    fea_index = 0
    # visual_scalars1 = combined_fea1[:,fea_index]
    # visual_scalars2 = combined_fea2[:,fea_index]
    #visual_scalars1 = (points_1-points_1.min(0))/(points_1.max(0)-points_1.min(0))
    visualize_point_fea(points_1, combined_fea1,rgb=True)
    #visualize_point_fea(points_2,visual_scalars2)

