from pykeops.torch import LazyTensor
from shapmagn.utils.obj_factory import obj_factory
from shapmagn.utils.local_feature_extractor import compute_anisotropic_gamma_from_points


def KNN(K):
    def compute(pc1, pc2):
        B,N = pc1.shape[0], pc1.shape[1]
        pc_i = LazyTensor(pc1[:,:,None])
        pc_j = LazyTensor(pc2[:,None])
        dist2 = pc_i.sqdist(pc_j)
        K_min, index=dist2.min_argmin(K=K, dim=2)
        return K_min, index.long().view(B, N, K)
    return compute


def AnisoKNN(K,compute_gamma_obj):
    compute_gamma = obj_factory(compute_gamma_obj)
    def compute(pc1, pc2):
        B,N = pc1.shape[0], pc1.shape[1]
        gamma = compute_gamma(pc1)
        gamma = LazyTensor(gamma.view(gamma.shape[0], gamma.shape[1], -1)[:, None])  # Bx1xMxD*D
        pc_i = LazyTensor(pc1[:,:,None])
        pc_j = LazyTensor(pc2[:,None])
        dist2 = (pc_i - pc_j) | gamma.matvecmult(pc_i - pc_j)
        K_min, index=dist2.min_argmin(K=K, dim=2)
        return K_min, index.long().view(B, N, K)
    return compute