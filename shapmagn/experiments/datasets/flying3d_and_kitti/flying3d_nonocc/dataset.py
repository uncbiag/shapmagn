import os
import glob
import numpy as np
from shapmagn.experiments.datasets.flying3d_and_kitti.generic import SceneFlowDataset


class FT3D(SceneFlowDataset):
    def __init__(self, root_dir, nb_points, mode):
        """
        Construct the FlyingThing3D datatset as in:
        Gu, X., Wang, Y., Wu, C., Lee, Y.J., Wang, P., HPLFlowNet: Hierarchical
        Permutohedral Lattice FlowNet for scene ﬂow estimation on large-scale 
        point clouds. IEEE Conf. Computer Vision and Pattern Recognition 
        (CVPR). pp. 3254–3263 (2019) 

        Parameters
        ----------
        root_dir : str
            Path to root directory containing the datasets.
        nb_points : int
            Maximum number of points in point clouds.
        mode : str
            'train': training dataset.
            
            'val': validation dataset.
            
            'test': test dataset

        """

        super(FT3D, self).__init__(nb_points)

        self.mode = mode
        self.root_dir = root_dir
        self.filenames = self.get_file_list()

    def __len__(self):

        return len(self.filenames)

    def get_file_list(self):
        """
        Find and filter out paths to all examples in the dataset. 
        
        """
        mode_backup = self.mode
        if self.mode == "debug":
            self.mode = "train"
        # Get list of filenames / directories
        if self.mode == "train" or self.mode == "val":
            pattern = "train/0*"
        elif self.mode == "test":
            pattern = "val/0*"
        else:
            raise ValueError("Mode " + str(self.mode) + " unknown.")
        filenames = glob.glob(os.path.join(self.root_dir, pattern))

        # Train / val / test split
        if self.mode == "train" or self.mode == "val":
            assert len(filenames) == 19640, "Problem with size of training set"
            ind_val = set(np.linspace(0, 19639, 2000).astype("int"))
            ind_all = set(np.arange(19640).astype("int"))
            ind_train = ind_all - ind_val
            assert (
                len(ind_train.intersection(ind_val)) == 0
            ), "Train / Val not split properly"
            filenames = np.sort(filenames)
            if self.mode == "train":
                filenames = filenames[list(ind_train)]
            elif self.mode == "val":
                filenames = filenames[list(ind_val)]
        else:
            assert len(filenames) == 3824, "Problem with size of test set"


        if mode_backup=="debug":
            filenames = filenames[:20]
            self.mode="debug"

        return list(filenames)

    def load_sequence(self, idx):
        """
        Load a sequence of point clouds.

        Parameters
        ----------
        idx : int
            Index of the sequence to load.

        Returns
        -------
        sequence : list(np.array, np.array)
            List [pc1, pc2] of point clouds between which to estimate scene 
            flow. pc1 has size n x 3 and pc2 has size m x 3.
            
        ground_truth : list(np.array, np.array)
            List [mask, flow]. mask has size n x 1 and pc1 has size n x 3. 
            flow is the ground truth scene flow between pc1 and pc2. mask is 
            binary with zeros indicating where the flow is not valid/occluded.

        """

        # Load data
        sequence = []  # [Point cloud 1, Point cloud 2]
        for fname in ["pc1.npy", "pc2.npy"]:
            pc = np.load(os.path.join(self.filenames[idx], fname))
            pc[..., 0] *= -1
            pc[..., -1] *= -1
            sequence.append(pc)
        ground_truth = [
            np.ones_like(sequence[0][:, 0:1]),
            sequence[1] - sequence[0],
        ]  # [Occlusion mask, flow]

        return sequence, ground_truth
