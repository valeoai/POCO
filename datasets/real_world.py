from torch_geometric.data import Dataset
from lightconvpoint.datasets.data import Data
import os
import numpy as np
import torch
import logging

class RealWorld(Dataset):

    def __init__(self, root, split="training", transform=None, filter_name=None, num_non_manifold_points=2048, dataset_size=None, **kwargs):
            
        super().__init__(root, transform, None)

        logging.info(f"Dataset  - Real World- {dataset_size}")


        self.root = os.path.join(self.root, "real_world")
        
        self.filenames = []
        with open(os.path.join(self.root, "testset.txt")) as f:
            content = f.readlines()
            content = [line.split("\n")[0] for line in content]
            content = [os.path.join(self.root, "03_meshes", line) for line in content]
            self.filenames += content
        self.filenames.sort()

        if dataset_size is not None:
            self.filenames = self.filenames[:dataset_size]

        logging.info(f"Dataset - len {len(self.filenames)}")

    def get_category(self, f_id):
        return self.filenames[f_id].split("/")[-2]

    def get_object_name(self, f_id):
        return self.filenames[f_id].split("/")[-1]

    def get_class_name(self, f_id):
        return self.metadata[self.get_category(f_id)]["name"]

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def _download(self): # override _download to remove makedirs
        pass

    def download(self):
        pass

    def process(self):
        pass

    def _process(self):
        pass

    def len(self):
        return len(self.filenames)


    def get_data_for_evaluation(self, idx):
        filename = self.filenames[idx]
        raise NotImplementedError
        data_shape = np.load(os.path.join(filename, "pointcloud.npz"))
        data_space = np.load(os.path.join(filename, "points.npz"))
        return data_shape, data_space

    def get(self, idx):
        """Get item."""
        filename = self.filenames[idx]

        filename = filename.replace("03_meshes", "04_pts")
        pts_shp = np.load(filename+".xyz.npy")

        pts_shp = torch.tensor(pts_shp, dtype=torch.float32)

        data = Data(x = torch.ones_like(pts_shp),
                    shape_id=idx, 
                    pos=pts_shp,
                    normal=None,
                    pos_non_manifold=torch.zeros((1,3), dtype=torch.float32), 
                    occupancies=None, #
                    )

        return data