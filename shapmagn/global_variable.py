import os
from shapmagn.shape.point_cloud import PointCloud
from shapmagn.shape.surface_mesh import SurfaceMesh, SurfaceMesh_Point
from shapmagn.shape.poly_line import PolyLine
#SHAPMAGN_PATH =os.path.abspath("/playpen-raid1/zyshen/proj/shapmagn/shapmagn")
SHAPMAGN_PATH =os.path.abspath("/home/zyshen/proj/shapmagn/shapmagn")
shape_type = "pointcloud"
SHAPE_POOL = {
    "pointcloud": PointCloud,
    "surfacemesh": SurfaceMesh,
    "surfacemesh_pointmode": SurfaceMesh_Point,
    "polyline": PolyLine,
}
Shape = SHAPE_POOL[shape_type]


from shapmagn.metrics.reg_losses import *

LOSS_POOL = {
    "current": CurrentDistance,
    "varifold": VarifoldDistance,
    "geomloss": GeomDistance,
    "l2": L2Distance,
    "localreg": LocalReg,
    "gmm": GMMLoss,
}


from shapmagn.datasets.custom_dataset import CustomDataset
from shapmagn.datasets.pair_dataset import RegistrationPairDataset

DATASET_POOL = {
    "custom_dataset": CustomDataset,
    "pair_dataset": RegistrationPairDataset,
}


from shapmagn.models_reg.model_lddmm import LDDMMOPT
from shapmagn.models_reg.model_discrete_flow import DiscreteFlowOPT
from shapmagn.models_reg.model_gradient_flow import GradientFlowOPT
from shapmagn.models_reg.model_probreg import ProRegOPT
from shapmagn.models_reg.model_prealign import PrealignOPT
from shapmagn.models_reg.model_deep_feature import DeepFeature
from shapmagn.models_reg.model_deep_flow import DeepDiscreteFlow
from shapmagn.models_reg.model_wasserstein_barycenter import WasserBaryCenterOPT
from shapmagn.models_general.model_deep_pred import DeepPredictor

MODEL_POOL = {
    "lddmm_opt": LDDMMOPT,
    "discrete_flow_opt": DiscreteFlowOPT,
    "prealign_opt": PrealignOPT,
    "gradient_flow_opt": GradientFlowOPT,
    "feature_deep": DeepFeature,
    "flow_deep": DeepDiscreteFlow,
    "discrete_flow_deep": DeepDiscreteFlow,
    "barycenter_opt": WasserBaryCenterOPT,
    "probreg_opt": ProRegOPT,
    "deep_predictor": DeepPredictor,
}


from shapmagn.shape.point_sampler import point_grid_sampler, point_uniform_sampler

SHAPE_SAMPLER_POOL = {
    "point_grid": point_grid_sampler,
    "point_uniform": point_uniform_sampler,
}
# INTERPOLATOR_POOL = {"point_kernel":nadwat_kernel_interpolator, "point_spline": spline_intepolator}
