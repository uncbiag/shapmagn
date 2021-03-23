from shapmagn.shape.point_cloud import PointCloud
from shapmagn.shape.surface_mesh import SurfaceMesh
from shapmagn.shape.poly_line import PolyLine
shape_type = "pointcloud"
SHAPE_POOL = {"pointcloud": PointCloud, "surfacemesh": SurfaceMesh, "polyline": PolyLine}
Shape = SHAPE_POOL[shape_type]


from shapmagn.metrics.losses import *
LOSS_POOL = {"current": CurrentDistance, "varifold": VarifoldDistance, "geomloss": GeomDistance, "l2": L2Distance, "localreg":LocalReg,"gmm":GMMLoss}


from shapmagn.datasets.reg_pair_dataset import RegistrationPairDataset
from shapmagn.debugging.reg_pair_synth_dataset import RegistrationPairSynthDataset
DATASET_POOL = {
    "pair_dataset": RegistrationPairDataset,
    "pair_synth_dataset": RegistrationPairSynthDataset
}


from shapmagn.models.model_lddmm import LDDMMOPT
from shapmagn.models.model_discrete_flow import DiscreteFlowOPT
from shapmagn.models.model_gradient_flow import GradientFlowOPT
from shapmagn.models.model_prealign import PrealignOPT
from shapmagn.models.model_deep_feature import DeepFeature
from shapmagn.models.model_deep_flow import DeepDiscreteFlow
MODEL_POOL = {"lddmm_opt": LDDMMOPT, "discrete_flow_opt": DiscreteFlowOPT, "prealign_opt": PrealignOPT,"gradient_flow_opt":GradientFlowOPT,
              "feature_deep":DeepFeature, "flow_deep":DeepDiscreteFlow,"discrete_flow_deep": DeepDiscreteFlow}


from shapmagn.shape.point_sampler import point_grid_sampler, point_uniform_sampler
SHAPE_SAMPLER_POOL = {"point_grid": point_grid_sampler, "point_uniform": point_uniform_sampler}
# INTERPOLATOR_POOL = {"point_kernel":nadwat_kernel_interpolator, "point_spline": spline_intepolator}
