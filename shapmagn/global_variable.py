from shapmagn.metrics.losses import *
from shapmagn.datasets.dataset import RegistrationDataset
from shapmagn.shape.point_cloud import PointCloud
from shapmagn.shape.surface_mesh import SurfaceMesh
from shapmagn.shape.poly_line import PolyLine
LOSS_POOL ={"current": CurrentDistance, "varifold":VarifoldDistance, "geomloss":GeomDistance, "l2":L2Distance}
DATASET_POOL = {
    "pair_dataset": RegistrationDataset
}
shape_type = "point_cloud"

SHAPE_POOL = {"point_cloud":PointCloud, "surface_mesh":SurfaceMesh, "poly_line":PolyLine}
Shape = SHAPE_POOL[shape_type]
