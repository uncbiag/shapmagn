import os
import glob
import numpy as np
from shapmagn.experiments.datasets.flying3d_and_kitti.generic import SceneFlowDataset


class FT3D(SceneFlowDataset):
    def __init__(self, root_dir, nb_points, mode):
        """
        Construct the FlyingThing3D datatset as in:
        Liu, X., Qi, C.R., Guibas, L.J.: FlowNet3D: Learning scene ﬂow in 3D 
        point clouds. IEEE Conf. Computer Vision and Pattern Recognition 
        (CVPR). pp. 529–537 (2019) 
        
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

        #
        mode_backup=self.mode
        if self.mode=="debug":
            self.mode="train"
        if self.mode == "train" or self.mode == "val":
            pattern = "TRAIN_*.npz"
        elif self.mode == "test":
            pattern = "TEST_*.npz"
        else:
            raise ValueError("Mode " + str(self.mode) + "unknown.")
        filenames = glob.glob(os.path.join(self.root_dir, pattern))

        # Remove one sample containing a nan value in train set
        scan_with_nan_value = os.path.join(
            self.root_dir, "TRAIN_C_0140_left_0006-0.npz"
        )
        if scan_with_nan_value in filenames:
            filenames.remove(scan_with_nan_value)

        # Remove samples with all points occluded in train set
        scan_with_points_all_occluded = [
            "TRAIN_A_0364_left_0008-0.npz",
            "TRAIN_A_0364_left_0009-0.npz",
            "TRAIN_A_0658_left_0014-0.npz",
            "TRAIN_B_0053_left_0009-0.npz",
            "TRAIN_B_0053_left_0011-0.npz",
            "TRAIN_B_0424_left_0011-0.npz",
            "TRAIN_B_0609_right_0010-0.npz",
        ]
        for f in scan_with_points_all_occluded:
            if os.path.join(self.root_dir, f) in filenames:
                filenames.remove(os.path.join(self.root_dir, f))

        # Remove samples with all points occluded in test set
        scan_with_points_all_occluded = [
            "TEST_A_0149_right_0013-0.npz",
            "TEST_A_0149_right_0012-0.npz",
            "TEST_A_0123_right_0009-0.npz",
            "TEST_A_0123_right_0008-0.npz",
        ]
        for f in scan_with_points_all_occluded:
            if os.path.join(self.root_dir, f) in filenames:
                filenames.remove(os.path.join(self.root_dir, f))

        # Train / val / test split
        if self.mode == "train" or self.mode == "val":
            ind_val = set(np.linspace(0, len(filenames) - 1, 2000).astype("int"))
            ind_all = set(np.arange(len(filenames)).astype("int"))
            ind_train = ind_all - ind_val
            assert (
                len(ind_train.intersection(ind_val)) == 0
            ), "Train / Val not split properly"
            filenames = np.sort(filenames)
            if self.mode == "train":
                filenames = filenames[list(ind_train)]
            elif self.mode == "val":
                filenames = filenames[list(ind_val)]


        if mode_backup=="debug":
            filenames = filenames[:20]
            self.mode="debug"

        return filenames

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
        with np.load(self.filenames[idx]) as data:
            sequence = [data["points1"], data["points2"]]
            ground_truth = [data["valid_mask1"].reshape(-1, 1), data["flow"]]

        return sequence, ground_truth
