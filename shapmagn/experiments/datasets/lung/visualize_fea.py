import numpy as np
import torch
import pykeops
# pykeops.clean_pykeops()
from pykeops.torch import LazyTensor
from pykeops.torch.cluster import grid_cluster, sort_clusters, cluster_ranges_centroids, from_matrix
import pyvista as pv

def numpy(x):
    return x.detach().cpu().numpy()



def local_moments(points, radius=1, ranges=None):
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




cloud_1 = pv.read("/playpen-raid1/Data/UNC_vesselParticles/10005Q_EXP_STD_NJC_COPD_wholeLungVesselParticles.vtk")
cloud_2 = pv.read("/playpen-raid1/Data/UNC_vesselParticles/10005Q_INSP_STD_NJC_COPD_wholeLungVesselParticles.vtk")
#print(cloud_1.array_names)


mask = cloud_1.points[:,0] < 0
points = cloud_1.points[mask]
points = points[:10000,:]

print(points.shape)

xyz = torch.from_numpy(points)


# Kernel truncation:
eps = 0
sigma = 1
xyz_labels = grid_cluster(xyz, eps)  # class labels
xyz_ranges, xyz_centroids, _  = cluster_ranges_centroids(xyz, xyz_labels)
xyz, xyz_labels = sort_clusters(xyz, xyz_labels)
D = ((xyz_centroids[...,:,None,:] - xyz_centroids[...,None,:,:])**2).sum(-1)
keep = D < (4 * sigma + 2 * eps)**2

for i in range(len(keep)):
    if not keep[i, i]:
        print(i)

print(f"We keep {100 * keep.sum() / (1.0*keep.shape[0] * keep.shape[1]):.2f}% of the {keep.shape} cluster-cluster matrix.")

ranges_xyz = from_matrix(xyz_ranges, xyz_ranges, keep)
#print(ranges_xyz)

mass, dev, cov = local_moments(xyz, radius = sigma, ranges=ranges_xyz)  # (N,), (N, D), (N, D, D)


order = 1

if order == 0:
    t = mass

elif order == 1:
    t = (dev ** 2).sum(-1)

else:
    vals = cov.symeig(eigenvectors=False).eigenvalues
    t = (mass**2) * vals[:,1] * vals[:,2]

t = t.clamp(0, 10).sqrt() * (mass > 2.5)

# Display
points  = numpy(xyz)
scalars = numpy(t)

plotter = pv.Plotter()
plotter.add_mesh(pv.PolyData(points), 
                 scalars = scalars, 
                 cmap = "magma", point_size=10,   
                 render_points_as_spheres=True,
                 opacity="linear",
                 lighting=True,
                 style="points", show_scalar_bar=True)
#plotter.add_point_labels([cloud_1.center,], ['Center',],
#                          point_color='yellow', point_size=20)
# plotter.show_grid()
# plotter.show()
data = pv.PolyData(points)
data.point_arrays['value'] = scalars
data.save("/playpen-raid1/zyshen/debug/debugg_point_visual2.vtk")

