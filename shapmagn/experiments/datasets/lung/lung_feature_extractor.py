
import math
import numpy as np
import torch
from pykeops.torch import LazyTensor
from shapmagn.utils.visualizer import visualize_point_fea,visualize_point_fea_with_arrow, visualize_point_overlap
from torchvectorized.vlinalg import vSymEig
# import pykeops
# pykeops.clean_pykeops()


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



def compute_aniso_local_moments(points, gamma=None):
    # B, N, D = points.shape
    shape_head, D = points.shape[:-1], points.shape[-1]

    x = points  # Normalize the kernel size
    # Computation:
    xp_i = LazyTensor(x[...,:,None,:])
    xp_j = LazyTensor(x[...,None,:,:])
    dist2 = xp_i.weightedsqdist(xp_j, gamma)
    K_ij = (-dist2).exp()  # BxNxN

    x = torch.cat((torch.ones_like(x[...,:1]), x), dim = -1)  # (B, N, D+1)

    x_i = LazyTensor(x[...,:,None,:])  # (B, N, 1, D+1)
    x_j = LazyTensor(x[...,None,:,:])  # (B, 1, N, D+1)

    C_ij = (K_ij * x_j).tensorprod(x_j)  # (B, N, N, (D+1)*(D+1))
    # here the dim should be 1,  self-centered mode  if set dim=2, then is the interpolate mode
    C_i  = C_ij.sum(dim =1).view(shape_head + (D+1, D+1))  # (B, N, D+1, D+1)

    w_i = C_i[...,:1,:1]               # (B, N, 1, 1), weights
    m_i = C_i[...,:1,1:]        # (B, N, 1, D), sum
    c_i = C_i[...,1:,1:] # (B, N, D, D), outer products

    mass_i = w_i.squeeze(-1)  # (B, N)
    dev_i = (m_i / w_i).squeeze(-2) - points  # (B, N, D)
    cov_i  = (c_i - (m_i.transpose(-1, -2) * m_i) / w_i) / w_i  # (B, N, D, D)

    return mass_i, dev_i, cov_i


def compute_local_fea_from_moments(fea_type, mass,dev,cov):
    fea = None
    if fea_type=="mass":
        fea = mass
    elif fea_type == "dev":
        fea = (dev ** 2).sum(-1,keepdim=True)
    elif 'eign' in fea_type:
        compute_eign_vector = fea_type in ["eignvector","eignvector_main"]
        B, N = cov.shape[0], cov.shape[1]
        cov = cov.view(B, N, -1, 1, 1).permute(0, 2, 1, 3, 4)
        vals, vectors = vSymEig(cov, eigenvectors=compute_eign_vector, flatten_output=True,descending_eigenvals=True)
        vals = vals.view(B, N, -1)
        if vectors is not None:
            vectors = vectors/torch.norm(vectors,p=2,dim=1,keepdim=True)  # BxNx1
            assert not torch.any(torch.isnan(vectors)) and not torch.any(torch.isinf(vectors))
        if fea_type == "eignvalue_prod":
            fea =  mass*vals[..., 1:2]* vals[..., 2:3]
        elif fea_type == "eignvalue_cat":
            fea = vals
        elif fea_type=="eignvector_main":
            vectors = vectors.view(B, N, 3, 3)
            fea = vectors[...,0]
            #fea = torch.sign(fea[:,:,0:1]) * fea
        elif fea_type=="eignvector":
            fea = vectors.view(B,N,-1)

    else:
        raise ValueError("not implemented")
    return fea




def feature_extractor(fea_type_list, weight_list=None, radius=1.,std_normalize=True, include_pos=False):
    def _compute_fea(points, return_stats=False, mean=None, std=None):
        if isinstance(points, np.ndarray):
            points =torch.from_numpy(points)
        nonlocal weight_list
        if weight_list is None:
            weight_list = [1.]* len(fea_type_list)
        mass, dev, cov = compute_local_moments(points, radius=radius)  # (N,), (N, D), (N, D, D)
        fea_list = [compute_local_fea_from_moments(fea_type, mass, dev, cov) for fea_type in fea_type_list]
        fea_dim_list = [fea.shape[-1] for fea in fea_list]
        weights = []
        for i, dim in enumerate(fea_dim_list):
            weights += [weight_list[i]/math.sqrt(dim)]*dim
        fea_combined = None
        if fea_dim_list:
            weights = torch.tensor(weights).view(1, 1, sum(fea_dim_list)).to(points.device)
            fea_combined = torch.cat(fea_list,-1)
            if std_normalize:
                if mean is None and std is None:
                    mean = fea_combined.mean(1,keepdim=True)
                    std = fea_combined.std(1,keepdim=True)
                fea_combined = (fea_combined-mean)/std
                fea_combined = fea_combined.clamp(-1,1)
                fea_combined = fea_combined*weights
            if include_pos:
                fea_combined = torch.cat([points,fea_combined],-1)
        if return_stats:
            return fea_combined, mean, std, mass
        else:
            return fea_combined, mass
    return _compute_fea







def pair_feature_extractor(fea_type_list,weight_list=None, radius=0.01, std_normalize=True, include_pos=False):
    fea_extractor = feature_extractor(fea_type_list,weight_list, radius, std_normalize, include_pos)
    def extract(flowed, target):
        flowed.pointfea, mean, std, _ = fea_extractor(flowed.points, return_stats=True)
        target.pointfea, _ = fea_extractor(target.points, mean=mean, std=std)
        return flowed, target
    return extract

