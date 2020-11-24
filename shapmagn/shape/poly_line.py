import torch

from shapmagn.shape.shape_base import ShapeBase

class PolyLine(ShapeBase):
    """
    This class is designed for batch based processing.
    2d polyline with edges
    For each batch, we assume nodes are sampled into the same size

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    # Constructor.
    def __init__(self):
        """

        :param points: BxNxD
        :param edges: BxNx2
        """
        super(PolyLine,self).__init__()
        self.type = 'polyline'
        self.edges =None
        self.index = None

    def set_data(self, points, weights=None, landmarks=None, pointfea=None, label=None, seg=None, edges=None, index=None, reindex=False):
        """

        :param points: BxNxD
        :param edges: BxNx2
        :param index: [index_a_list, index_b_list], each is an overbatch index list with B*N length
        :param reindex: generate index over batch for two ends
        :return:
        """
        self.points = points
        self.weights = weights
        self.landmarks = landmarks
        self.pointfea = pointfea
        self.label = label
        self.seg = seg
        assert edges is not None
        self.edges = edges
        if index is not None:
            self.index = index
        if self.index is None or reindex:
            index_a_list = []
            index_b_list = []
            for b in range(self.batch):
                index_a_list += edges[b,0]+ b*self.npoints
                index_b_list += edges[b,1]+ b*self.npoints
            self.index = [index_a_list, index_b_list]
        self.update_info()


    def set_data_with_refer_to(self, points, polyline):
        self.points = points
        self.edges = polyline.edges
        self.index= polyline.index
        self.label = polyline.label
        self.name_list = polyline.name_list
        self.landmarks = polyline.landmarks
        self.pointfea = polyline.pointfea
        self.weights = polyline.weights
        self.seg = polyline.seg
        self.update_info()



    def get_edges(self):
        return self.edges



    def get_centers_and_currents(self):
        """

        :return: centers:BxNxD, currents: BxNxD
        """

        a = self.points.view(-1)[self.index[0]]
        b = self.points.view(-1)[self.index[1]]
        centers = (a + b) / 2.
        currents = b - a
        zero_normal_index = torch.nonzero(torch.norm(currents, 2, 2) == 0)
        if zero_normal_index.shape[0] > 0:
            currents.data[zero_normal_index] = 1e-7
            print(" {} zero normal is detected, set the zero value to 1e-7".format(len(zero_normal_index)))
        return centers.view([self.batch,-1, self.dimension]), currents.view([self.batch,-1, self.dimension])