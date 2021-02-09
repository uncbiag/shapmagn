import logging
logger = logging.getLogger(__name__)
from shapmagn.shape.shape_base import ShapeBase

class PointCloud(ShapeBase):
    """
    This class is designed for batch based processing.
    For each batch, we assume the point clouds are sampled into the same size

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    # Constructor.
    def __init__(self):
        """

        :param points: BxNxD
        """
        super(PointCloud,self).__init__()
        self.type = 'pointcloud'
        self.points_mode_on = True


    def set_data_with_refer_to(self, points, pointcloud):
        self.points = points
        self.label = pointcloud.label
        self.name_list = pointcloud.name_list
        self.landmarks = pointcloud.landmarks
        self.pointfea = pointcloud.pointfea
        self.weights = pointcloud.weights
        self.seg = pointcloud.seg
        self.scale = pointcloud.scale
        self.update_info()
        return self

    def get_centers(self):
        """

        :return: centers:BxNxD
        """
        centers = self.points
        return centers
