import os
import numpy as np
import glob
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import logging



class ShapeNetSyntheticRooms(Dataset):

    def __init__(self, root, split="training", transform=None, filter_name=None, num_non_manifold_points=2048, dataset_size=None, **kwargs):

        super().__init__(root, transform, None)

        logging.info(f"ShapeNetSyntheticRoom")

        input_directories = ["rooms_04", "rooms_05", "rooms_06", "rooms_07", "rooms_08"]
        self.split = split
        self.filter_name = filter_name
        self.num_non_manifold_points = num_non_manifold_points

        self.filenames = []
        for input_directory in input_directories:
            if self.split in ["training", "train"]:
                split_file = ["train"]
            elif self.split in ["test", "testing"]:
                split_file = ["test"]
            elif self.split in ["val", "validation"]:
                split_file = ["val"]
            elif self.split in ["trainval", "trainVal", "TrainVal"]:
                split_file = ["train", "val"]
            else:
                raise ValueError(f"Wrong split value {self.split}")
            for sp_file in split_file:
                lines = open(os.path.join(self.root, input_directory, f"{sp_file}.lst")).readlines()
                lines = [l.split("\n")[0] for l in lines]
                lines = [os.path.join(self.root, input_directory, l) for l in lines]
                self.filenames += lines

        if dataset_size is not None:
            self.filenames = self.filenames[:dataset_size]
        logging.info(f"dataset len {len(self.filenames)}")


        
        self.object_classes = ['04256520', '03636649', '03001627', '04379243', '02933112']
        self.object_classes.sort()

        self.class_corresp = {
            0: "outside",
            1: "ground",
            2: "wall",
            3:'02933112',
            4:'03001627',
            5: '03636649',
            6: '04256520',
            7: '04379243',
        }

        self.class_colors = {
            1: [100,100,100],
            2: [255,255,0],
            3: [255,0,0],
            4: [0,255,0],
            5: [0,0,255],
            6: [255,0,255],
            7: [0,255,255],
        }

    def get_category(self, f_id):
        return self.filenames[f_id].split("/")[-2]

    def get_object_name(self, f_id):
        return self.filenames[f_id].split("/")[-1]

    def get_class_name(self, f_id):
        return self.filenames[f_id].split("/")[-2]

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def _download(self): 
        pass

    def download(self):
        pass

    def _process(self):
        pass

    def process(self):
        pass

    def len(self):
        return len(self.filenames)

    def get_data_for_evaluation(self, idx):
        scene = self.filenames[idx]

        input_pointcloud = glob.glob(os.path.join(scene, "pointcloud/*.npz"))
        input_pointcloud = input_pointcloud[torch.randint(0,len(input_pointcloud),size=(1,)).item()]
        input_pointcloud = np.load(input_pointcloud)

        non_manifold_pc = glob.glob(os.path.join(scene, "points_iou/*.npz"))
        non_manifold_pc = non_manifold_pc[torch.randint(0,len(non_manifold_pc),size=(1,)).item()]
        non_manifold_pc = np.load(non_manifold_pc)

        return input_pointcloud, non_manifold_pc

    def get(self, idx):
        """Get item."""

        scene = self.filenames[idx]

        manifold_data = glob.glob(os.path.join(scene, "pointcloud/*.npz"))
        manifold_data = manifold_data[torch.randint(0,len(manifold_data),size=(1,)).item()]
        manifold_data = np.load(manifold_data)
        points_shape = manifold_data["points"]
        normals_shape = manifold_data["normals"]
        pts_shp = torch.tensor(points_shape, dtype=torch.float)
        nls_shp = torch.tensor(normals_shape, dtype=torch.float)

        
        non_manifold_data = glob.glob(os.path.join(scene, "points_iou/*.npz"))
        non_manifold_data = non_manifold_data[torch.randint(0,len(non_manifold_data),size=(1,)).item()]
        non_manifold_data = np.load(non_manifold_data)
        points_space = torch.tensor(non_manifold_data["points"], dtype=torch.float)
        occupancies = torch.tensor(np.unpackbits(non_manifold_data['occupancies']), dtype=torch.long)


        data = Data(x = torch.ones_like(pts_shp),
                    shape_id=idx, 
                    pos=pts_shp,
                    normal=nls_shp,
                    pos_non_manifold=points_space, occupancies=occupancies, #
                    )

        return data