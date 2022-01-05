from unicodedata import category
import torch
import numpy as np
import lightconvpoint.utils.data_utils as data_utils
import os

from torch_geometric.data import Dataset
from lightconvpoint.datasets.data import Data
import logging

import importlib
if importlib.util.find_spec("valeodata") is not None:
    valeodata_exists = True
    import valeodata
else:
    valeodata_exists = False




class ShapeNet(Dataset):

    def __init__(
        self, root, split="training",transform=None):
        super().__init__(root, transform, None)
        self.split = split
        self.num_classes = 50
        self.label_names = [
            ["Airplane", 4],
            ["Bag", 2],
            ["Cap", 2],
            ["Car", 4],
            ["Chair", 4],
            ["Earphone", 3],
            ["Guitar", 3],
            ["Knife", 2],
            ["Lamp", 4],
            ["Laptop", 2],
            ["Motorbike", 6],
            ["Mug", 2],
            ["Pistol", 3],
            ["Rocket", 3],
            ["Skateboard", 3],
            ["Table", 3],
        ]

        self.category_range = []
        count = 0
        for element in self.label_names:
            part_start = count
            count += element[1]
            part_end = count
            self.category_range.append([part_start, part_end])

        if self.split == 'training':
            filelist_train = os.path.join(self.root, "train_files.txt")
            filelist_val = os.path.join(self.root, "val_files.txt")
            (
                data_train,
                labels_shape_train,
                data_num_train,
                labels_pts_train,
                _,
            ) = data_utils.load_seg(filelist_train)
            (
                data_val,
                labels_shape_val,
                data_num_val,
                labels_pts_val,
                _,
             ) = data_utils.load_seg(filelist_val)
            self.data = np.concatenate([data_train, data_val], axis=0)
            self.labels_shape = np.concatenate([labels_shape_train, labels_shape_val], axis=0)
            self.data_num = np.concatenate([data_num_train, data_num_val], axis=0)
            self.labels_pts = np.concatenate([labels_pts_train, labels_pts_val], axis=0)


        elif self.split == 'test':
            filelist_test = os.path.join(self.root, "test_files.txt")
            (
                data_test,
                labels_shape_test,
                data_num_test,
                labels_pts_test,
                _,
            ) = data_utils.load_seg(filelist_test)
            self.data = data_test
            self.labels_shape = labels_shape_test
            self.data_num = data_num_test
            self.labels_pts = labels_pts_test


    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def _download(self): # override _download to remove makedirs
        self.download()

    def download(self):
        logging.debug("ShapeNet dataset download")
        if valeodata_exists:
            self.root = valeodata.download(self.root)
        else:
            logging.info(f"Dataset at {self.root}")
        logging.debug(f"Dataset at {self.root}")

    def process(self):
        pass

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        """Get item."""
        # get the data
        npts = self.data_num[idx]
        shape_label = int(self.labels_shape[idx])

        # the points and target
        pos = torch.tensor(self.data[idx, :npts], dtype=torch.float)
        y = torch.tensor(self.labels_pts[idx, :npts], dtype=torch.long)
        x = torch.ones_like(pos)
        category_filter = torch.zeros(self.num_classes, dtype=torch.float)
        part_start, part_end = self.category_range[shape_label]
        category_filter[part_start: part_end] = 1
        category_filter = category_filter
        data = Data(pos=pos, y=y, x=x,
                shape_id=idx, shape_label=shape_label,
                category_filter=category_filter)
        return data

    # def get_weights(self):
        
    #     if self.split == 'training':
    #         frequences = [0 for i in range(len(self.label_names))]
    #         for i in range(len(self.label_names)):
    #             frequences[i] += (self.labels_shape == i).sum()
    #         for i in range(len(self.label_names)):
    #             frequences[i] /= self.label_names[i][1]
    #         frequences = np.array(frequences)
    #         frequences = frequences.mean() / frequences
    #         repeat_factor = [sh[1] for sh in self.label_names]
    #         weights = np.repeat(frequences, repeat_factor)
    #     else:
    #         weights = None
    #     return weights