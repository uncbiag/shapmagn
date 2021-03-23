import os, sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))
import numpy as np
import torch
from shapmagn.utils.module_parameters import ParameterDict
from shapmagn.utils.obj_factory import obj_factory
from shapmagn.datasets.data_utils import get_file_name, generate_pair_name, get_obj
from shapmagn.shape.shape_pair_utils import create_shape_pair
from shapmagn.models.multiscale_optimization import build_single_scale_model_embedded_solver, build_multi_scale_solver
from shapmagn.global_variable import MODEL_POOL,Shape, shape_type
from shapmagn.utils.utils import get_grid_wrap_points
from shapmagn.utils.visualizer import *
from shapmagn.demos.demo_utils import *
from shapmagn.experiments.datasets.lung.lung_data_analysis import *


import matplotlib.pyplot as plt
import numpy as np

# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from pykeops.numpy import LazyTensor
import pykeops.config
# import pykeops
# pykeops.clean_pykeops()
dtype = "float32"  # No need for double precision here!
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags
from scipy.sparse.linalg.interface import IdentityOperator

# set shape_type = "pointcloud"  in global_variable.py
assert shape_type == "pointcloud", "set shape_type = 'pointcloud'  in global_variable.py"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
server_path = "/playpen-raid1/zyshen/proj/shapmagn/shapmagn/demos/" # "/playpen-raid1/"#"/home/zyshen/remote/llr11_mount/"
source_path =  server_path+"data/lung_vessel_demo_data/10031R_EXP_STD_NJC_COPD_wholeLungVesselParticles.vtk"
target_path = server_path + "data/lung_vessel_demo_data/10031R_INSP_STD_NJC_COPD_wholeLungVesselParticles.vtk"
compute_on_half_lung = True

####################  prepare data ###########################
pair_name = generate_pair_name([source_path,target_path])
reader_obj = "lung_dataloader_utils.lung_reader()"
scale = -1 # an estimation of the physical diameter of the lung, set -1 for auto rescaling   #[99.90687, 65.66011, 78.61013]
normalizer_obj = "lung_dataloader_utils.lung_normalizer(scale={})".format(scale)
sampler_obj = "lung_dataloader_utils.lung_sampler(method='voxelgrid',scale=0.01)"
get_obj_func = get_obj(reader_obj,normalizer_obj,sampler_obj, device)
source_obj, source_interval = get_obj_func(source_path)
target_obj, target_interval = get_obj_func(target_path)
min_interval = min(source_interval,target_interval)
input_data = {"source":source_obj,"target":target_obj}
create_shape_pair_from_data_dict = obj_factory("shape_pair_utils.create_source_and_target_shape()")
source, target = create_shape_pair_from_data_dict(input_data)
source = get_half_lung(source,normalize_weight=True) if compute_on_half_lung else source
target = get_half_lung(target,normalize_weight=True) if compute_on_half_lung else target






t = source.weights.cpu().squeeze().numpy()
x = source.points.cpu().squeeze().numpy()
N = x.shape[0]
x = x.astype(dtype)

################################################################
# To **display** our toy dataset with the (not-so-efficient) PyPlot library,
# we pick **10,000 points** at random:

N_display = N
indices_display = np.random.randint(0, N, N_display)

_, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), subplot_kw=dict(projection="3d"))
x_ = x[indices_display, :]
ax.scatter3D(x_[:, 0], x_[:, 1], x_[:, 2], c=t[indices_display], cmap=plt.cm.Spectral)
ax.set_title("{:,} out of {:,} points in our source point cloud".format(N_display, N))
plt.show()

####################################################################
# **Can we scale the spectral analysis presented above to this huge dataset?**
#
# In practice, the radius :math:`\sigma` of our
# kernel "adjacency" function is often **much smaller than the diameter of the input point cloud**:
# spectral methods rely on *small-range* neighborhoods to build
# a *global* coordinate system.
# Since :math:`k(x,y) \simeq 0` above a threshold of, say, :math:`4\sigma`,
# a simple way of accelerating
# the kernel-vector product :math:`v\mapsto K_{xx}v` in the (soft-)graph Laplacian is thus to
# **skip computations** between pairs of points that are far away from each other.
#
# As explained in :doc:`the documentation <../../python/sparsity>`,
# fast GPU routines rely heavily on **memory contiguity**:
# before going any further, we must
# **sort our input dataset** to make sure that neighboring points are stored
# next to each other on the device memory. As detailed in the
# :doc:`KeOps+NumPy tutorial on block-sparse reductions <../../_auto_examples/numpy/plot_grid_cluster_numpy>`,
# a simple way of doing so is to write:

# Import the KeOps helper routines for block-sparse reductions:
from pykeops.numpy.cluster import (
    grid_cluster,
    cluster_ranges_centroids,
    sort_clusters,
    from_matrix,
)

# Put our points in cubic bins of size eps, as we compute a vector of class labels:
eps = 0.001
x_labels = grid_cluster(x, eps)
# Compute the memory footprint and centroid of each of those non-empty "cubic" clusters:
x_ranges, x_centroids, _ = cluster_ranges_centroids(x, x_labels)
# Sort our dataset according to the vector of labels:
x, x_labels = sort_clusters(x, x_labels)

#############################################################################
#
# .. note::
#   In higher-dimensional settings, the simplistic
#   :func:`grid_cluster <pykeops.numpy.cluster.grid_cluster>`
#   scheme could be replaced by a more versatile routine such as
#   our :doc:`KeOps+NumPy K-means implementation <../kmeans/plot_kmeans_numpy>`.
#
# Points are now roughly sorted
# according to their locations, with each cluster corresponding to
# a contiguous slice of the (sorted) **x** array:

_, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), subplot_kw=dict(projection="3d"))
x_ = x[indices_display, :]
ax.scatter3D(x_[:, 0], x_[:, 1], x_[:, 2], c=x_labels[indices_display], cmap="prism")
ax.set_title("Cluster labels")
plt.show()

############################################################################
# We can prune computations out of the :math:`v\mapsto K_{xx} v`
# matrix-vector product in a GPU-friendly way by **skipping whole blocks**
# of cluster-cluster interactions.
# A good rule of thumb is to **only consider pairs of points** belonging
# to clusters :math:`X` and :math:`Y` whose centroids :math:`x_c` and
# :math:`y_c` are such that:
#
# .. math::
#   \|x_c - y_c\|^2 < \big( 4\sigma + \tfrac{1}{2}\text{diam}(X)  + \tfrac{1}{2}\text{diam}(Y) \big)^2.
#
# Considering that our cubic bins of size :math:`\varepsilon` all have a
# diameter that is equal to :math:`\sqrt{3}\,\varepsilon`, this "block-sparsity"
# pattern can be encoded in a small boolean matrix **keep** computed through:

sigma = (
    0.001 if pykeops.config.gpu_available else 0.1
)  # Standard deviation of our Gaussian kernel
# Compute a coarse Boolean mask:
D = np.sum((x_centroids[:, None, :] - x_centroids[None, :, :]) ** 2, 2)
keep = D < (4 * sigma + np.sqrt(3) * eps) ** 2

###############################################################
# which can then be converted to a GPU-friendly,
# `LIL-like sparsity pattern <https://en.wikipedia.org/wiki/Sparse_matrix#List_of_lists_(LIL)>`_
# with the :func:`from_matrix <pykeops.numpy.cluster.from_matrix>` helper:

ranges_ij = from_matrix(x_ranges, x_ranges, keep)

############################################################################
# Now, leveraging this information with KeOps is as simple
# as typing:

x_ = x / sigma  # N.B.: x is a **sorted** list of points
x_i, x_j = LazyTensor(x_[:, None, :]), LazyTensor(x_[None, :, :])
K_xx = (-((x_i - x_j) ** 2).sum(2) / 2).exp()  # Symbolic (N,N) Gaussian kernel matrix

K_xx.ranges = ranges_ij  # block-sparsity pattern
print(K_xx)

############################################################################
# A straightforward computation shows that our new
# **block-sparse** operator may be **up to 20 times more efficient** than a
# full KeOps :class:`pykeops.torch.LazyTensor`:

# Compute the area of each rectangle "cluster-cluster" tile in the full kernel matrix:
areas = (x_ranges[:, 1] - x_ranges[:, 0])[:, None] * (x_ranges[:, 1] - x_ranges[:, 0])[
    None, :
]
total_area = np.sum(areas)  # should be equal to N**2 = 1e12
sparse_area = np.sum(areas[keep])

print(
    "We keep {:.2e}/{:.2e} = {:2d}% of the original kernel matrix.".format(
        sparse_area, total_area, int(100 * sparse_area / total_area)
    )
)

############################################################################
# Good. Once we're done with these pre-processing steps,
# block-sparse :class:`pykeops.torch.LazyTensor` are just as easy to interface with **scipy** as
# regular NumPy arrays:

K = aslinearoperator(K_xx)

##########################################################################
# The normalized graph Laplacian can be defined as usual:

D = K @ np.ones(N, dtype=dtype)  # Sum along the lines of the adjacency matrix
D_2 = aslinearoperator(diags(1 / np.sqrt(D)))
L_norm = IdentityOperator((N, N)) - D_2 @ K @ D_2
L_norm.dtype = np.dtype(
    dtype
)  # Scipy Bugfix: by default, "-" removes the dtype information...

##########################################################################
# And our favourite solver will compute, as expected,
# the smallest eigenvalues of this custom operator:


from time import time

start = time()

# Compute the 7 smallest eigenvalues/vectors of our normalized graph Laplacian
eigenvalues, coordinates = eigsh(L_norm, k=7, which="SM")

print(
    "Smallest eigenvalues of the normalized graph Laplacian, computed in {:.3f}s ".format(
        time() - start
    )
    + "on a cloud of {:,} points in dimension {}:".format(x.shape[0], x.shape[1])
)
print(eigenvalues)

##########################################################################
#
# .. note::
#   On very large problems, a custom eigenproblem solver
#   implemented with the **PyTorch+KeOps** interface should be sensibly **faster**
#   than this SciPy wrapper: performing all computations on the GPU
#   would allow us to perform linear operations in parallel
#   and to **skip hundreds of unnecessary Host-Device memory transfers**.
#
# Anyway. Displayed on a subsampled point cloud (for the sake of efficiency),
# our spectral coordinates look good!

x_ = x[indices_display, :]

# sphinx_gallery_thumbnail_number = 5
_, axarr = plt.subplots(
    nrows=2, ncols=3, figsize=(12, 8), subplot_kw=dict(projection="3d")
)

for i in range(2):
    for j in range(3):
        axarr[i][j].scatter3D(
            x_[:, 0],
            x_[:, 1],
            x_[:, 2],
            c=coordinates[indices_display, 3 * i + j],
            cmap=plt.cm.Spectral,
        )
        axarr[i][j].set_title(
            "Eigenvalue {} = {:.1e}".format(3 * i + j + 1, eigenvalues[3 * i + j])
        )

plt.tight_layout()
plt.show()