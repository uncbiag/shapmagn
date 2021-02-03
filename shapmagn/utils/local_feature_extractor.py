
import math
import numpy as np
import torch
from pykeops.torch import LazyTensor
from shapmagn.utils.visualizer import visualize_point_fea,visualize_point_fea_with_arrow, visualize_point_overlap
from torchvectorized.vlinalg import vSymEig
from shapmagn.shape.point_interpolator import nadwat_kernel_interpolator,ridge_kernel_intepolator
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
    gamma = gamma.view(gamma.shape[0], gamma.shape[1], -1)
    # B, N, D = points.shape
    shape_head, D = points.shape[:-1], points.shape[-1]

    x = points  # Normalize the kernel size
    # Computation:
    xp_i = LazyTensor(x[...,:,None,:])
    xp_j = LazyTensor(x[...,None,:,:])
    dist2 = xp_i.weightedsqdist(xp_j, gamma)
    K_ij = (-dist2).exp()  # BxNxN

    x = torch.cat((torch.ones_like(x[...,:1]), x), dim = -1)  # (B, N, D+1)
    x_j = LazyTensor(x[...,None,:,:])  # (B, 1, N, D+1)

    C_ij = (K_ij * x_j).tensorprod(x_j)  # (B, N, N, (D+1)*(D+1))
    # if  dim= 1,  self-centered mode  if set dim=2, then is the interpolate mode
    C_i  = C_ij.sum(dim =2).view(shape_head + (D+1, D+1))  # (B, N, D+1, D+1)

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
    elif 'eigen' in fea_type:
        compute_eigen_vector = fea_type in ["eigenvector","eigenvector_main"]
        B, N, D = cov.shape[0], cov.shape[1], cov.shape[-1]
        cov = cov.view(B, N, -1, 1, 1).permute(0, 2, 1, 3, 4)
        vals, vectors = vSymEig(cov, eigenvectors=compute_eigen_vector, flatten_output=True,descending_eigenvals=True)
        vals = vals.view(B, N, -1)
        if vectors is not None:
            vectors = vectors+1e-9 # avoid divide by 0
            vectors = vectors/torch.norm(vectors,p=2,dim=1,keepdim=True)  # BxNx1
            assert not torch.any(torch.isnan(vectors)) and not torch.any(torch.isinf(vectors))
        if fea_type == "eigenvalue_prod":
            if D==2:
                fea = mass*vals[..., 1:2]
            elif D==3:
                fea =  mass*vals[..., 1:2]* vals[..., 2:3]
        elif fea_type == "eigenvalue_cat":
            fea = vals
        elif fea_type=="eigenvector_main":
            vectors = vectors.view(B, N, D, D)
            fea = vectors[...,0]
        elif fea_type=="eigenvector":
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


def get_Gamma(sigma_scale, principle_weight= None, eigenvalue=None, eigenvector=None,):
    """

    :param sigma_scale: scale the sigma
    :param principle_weight: a list of sigma of D size, e.g.[1,1,1]
    :param eigenvalue: torch.Tensor, BXNxD, optional if the eigenvalue is not None, then the principle vector = normalized(eigenvale)*sigma_scale
    :param eigenvector: torch.Tensor, BXNxDxD, denote as U
    :return: Gamma: torch.Tensor, BxNxDxD  U{\Lambda}^{-2}U^T
    """
    device = eigenvector.device
    nbatch, npoints = eigenvector.shape[0], eigenvector.shape[1]
    if eigenvalue is not None:
        principle_weight = eigenvalue/torch.norm(eigenvalue,p=2,dim=2,keepdim=True)*sigma_scale
        principle_weight_inv_sq = 1/(principle_weight**2)
        principle_diag = torch.diag_embed(principle_weight_inv_sq)
    else:

        principle_weight = np.array(principle_weight)/np.linalg.norm(principle_weight)*sigma_scale
        principle_weight = principle_weight.astype(np.float32)
        principle_weight_inv_sq = 1/(principle_weight**2)
        principle_diag = torch.diag(torch.tensor(principle_weight_inv_sq).to(device)).repeat(nbatch,npoints,1,1) # BxNxDxD
        principle_weight =torch.tensor(principle_weight).to(device).repeat(1,npoints,1)
    Gamma = eigenvector @ principle_diag @ (eigenvector.permute(0,1,3,2))
    return Gamma,principle_weight


def compute_anisotropic_gamma_from_points(points,cov_sigma_scale=0.05,aniso_kernel_scale=None,principle_weight=None,eigenvalue_min=0.1):
    """
    compute inverse covariance matrix for anisotropic kernel
    this function doesn't support auto-grad
    :param input_shape: torch.Tensor, BxNxD
    :param cov_sigma_scale: float, the sigma used for computing the local covariance matrix for eigenvalue, eigenvector extraction
    :param aniso_kernel_scale: float,  anisotropic kernel scale
    :param principle_weight: list of size D, weight of directions in anistropic kernel, don't have to be norm to 1, will normalized later, if not given, use the eigenvalue instead
    :param eigenvalue_min: float, if the principal vector is not given, then the norm2 normalized eigenvalue will be used for compute the weight of each principle direction,
     this value is to control the weight of the eigenvector, to avoid extreme narraw direction (typically happens when eigenvalue close to zero)
    :return: Gamma, torch.Tensor, BxNxDxD  U{\Lambda}^{-2}U^T,  where U is the eigenvector of the local covariance matrix
    """
    aniso_kernel_scale = aniso_kernel_scale if aniso_kernel_scale is not None else cov_sigma_scale
    B, D = points.shape[0], points.shape[-1]
    fea_type_list = ["eigenvalue_cat", "eigenvector"]
    fea_extractor = feature_extractor(fea_type_list, radius=cov_sigma_scale, std_normalize=False, include_pos=False)
    combined_fea, mass = fea_extractor(points)
    eigenvalue, eigenvector = combined_fea[:, :, :D], combined_fea[:, :, D:]
    eigenvector = eigenvector.view(eigenvector.shape[0], eigenvector.shape[1], D, D)
    print("detect there is {} eigenvalue smaller or equal to 0, set to 1e-7".format(torch.sum(eigenvalue <= 0)))
    if principle_weight is None:
        eigenvalue[eigenvalue <= 0.] = 1e-7
        eigenvalue = eigenvalue / torch.norm(eigenvalue, p=2, dim=2, keepdim=True)
        eigenvalue[eigenvalue < eigenvalue_min] = eigenvalue_min
    else:
        eigenvalue = None
    Gamma, principle_weight = get_Gamma(aniso_kernel_scale, principle_weight=principle_weight, eigenvalue=eigenvalue,
                                        eigenvector=eigenvector)
    return Gamma.view(Gamma.shape[0],Gamma.shape[1],D,D), principle_weight



def nadwat_interpolator_with_aniso_kernel_extractor_embedded(interpolator_setting):
    exp_order=interpolator_setting[("exp_order", 2,"exp order when computing distance")]
    cov_sigma_scale = interpolator_setting[("cov_sigma_scale", 0.05,"the sigma used for computing the local covariance matrix for eigenvalue, eigenvector extraction")]
    aniso_kernel_scale = interpolator_setting[("aniso_kernel_scale", 0.05,"anisotropic kernel size")]
    principle_weight = interpolator_setting[("principle_weight", [2.,1.,1.],"list of size D, weight of directions in anistropic kernel, don't have to be norm to 1, will normalized later")]
    eigenvalue_min = interpolator_setting[("eigenvalue_min", 0.1,"the min value of the eigenvalue")]
    interp = nadwat_kernel_interpolator(exp_order=exp_order, iso=False)

    def compute(points,control_points,control_value,control_weights, gamma=None):
        Gamma_control_points = gamma
        if Gamma_control_points is None:
            Gamma_control_points, principle_weight_control_points = compute_anisotropic_gamma_from_points(points,
                                                                                          cov_sigma_scale=cov_sigma_scale,
                                                                                          aniso_kernel_scale=aniso_kernel_scale,
                                                                                          principle_weight=principle_weight,
                                                                                          eigenvalue_min=eigenvalue_min)

        interp_value = interp(points, control_points,control_value,control_weights,Gamma_control_points)
        return interp_value
    return compute


