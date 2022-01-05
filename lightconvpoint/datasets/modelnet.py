import numpy as np
import torch
import os
import pandas
import h5py

from torch_geometric.data import Dataset#, Data
from lightconvpoint.datasets.data import Data
import logging

import importlib
if importlib.util.find_spec("valeodata") is not None:
    valeodata_exists = True
    import valeodata
else:
    valeodata_exists = False

class Modelnet40_ply_hdf5_2048(Dataset):

    def load_data(self,files):

        train_filenames = []
        for line in open(os.path.join(self.root, files)):
            line = line.split("\n")[0]
            line = os.path.basename(line)
            train_filenames.append(os.path.join(self.root, line))

        data = []
        labels = []
        for filename in train_filenames:
            f = h5py.File(filename, "r")
            data.append(f["data"])
            labels.append(f["label"])

        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)

        return data, labels

    def __init__(self, root, split="training",transform=None, use_normals=False):
        super(Modelnet40_ply_hdf5_2048, self).__init__(root, transform, None)
        self.split=split

        self.use_normals = use_normals
        if self.use_normals:
            logging.warning("This version of the dataset does not include normals")

        # get the data
        if self.split == 'training':
            self.data, self.labels = self.load_data("train_files.txt")
        elif self.split == 'test':
            self.data, self.labels = self.load_data("test_files.txt")

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def _download(self): # override _download to remove makedirs
        self.download()

    def download(self):
        logging.debug("ModelNet dataset download")
        if valeodata_exists:
            self.root = valeodata.download(self.root)
        else:
            logging.info(f"Dataset at {self.root}")
        logging.debug(f"Dataset at {self.root}")

    def _process(self):
        pass

    def process(self):
        pass

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        """Get item."""

        # the points and target
        pos = torch.tensor(self.data[idx], dtype=torch.float)
        x = torch.ones((pos.shape[0],3))
        y = int(self.labels[idx])
        data = Data(pos=pos, y=y, x=x, shape_id=idx)
        return data

class Modelnet40_normal_resampled(Dataset):

    def __init__(self, root, split="training",transform=None, in_memory=False, use_normals=False):
        super(Modelnet40_normal_resampled, self).__init__(root, transform, None)
        self.split=split
        self.in_memory = in_memory
        self.use_normals = use_normals

        # load in memory
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))  

        # get the filepath
        shape_ids = {}
        if self.split == "training":
            shape_ids['training'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))] 
        elif self.split == "test":
            shape_ids['test']= [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]
        else:
            raise ValueError("Unknown split name")
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i])+'.txt') for i in range(len(shape_ids[split]))]

        if self.in_memory:
            # load everything in memory
            self.data = []
            self.labels = []
            for fname in self.datapath:
                self.data.append(pandas.read_csv(fname[1], header=0).values.astype(np.float32))
                self.labels.append(int(self.cat.index(fname[0])))
            self.data = np.stack(self.data, axis=0)
            self.labels = np.array(self.labels, dtype=np.int64)


    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def _download(self): # override _download to remove makedirs
        self.download()

    def download(self):
        logging.debug("ModelNet dataset download")
        if valeodata_exists:
            self.root = valeodata.download(self.root)
        else:
            logging.info(f"Dataset at {self.root}")
        logging.debug(f"Dataset at {self.root}")


    def process(self):
        pass

    def len(self):
        if self.in_memory:
            return self.data.shape[0]
        else:
            return len(self.datapath)


    def get_targets(self):
        targets = []
        for d in self.datapath:
            targets.append(self.cat.index(d[0]))
        return np.array(targets, dtype=np.int64)

    def get(self, idx):
        """Get item."""

        if self.in_memory:

            # the points and target
            pos = torch.tensor(self.data[idx][:,:3], dtype=torch.float)
            if self.use_normals:
                x = torch.tensor(self.data[idx][:,3:], dtype=torch.float)
            else:
                x = torch.ones((pos.shape[0],3))
            y = int(self.labels[idx])
            data = Data(pos=pos, y=y, x=x, shape_id=idx)
            return data

        else:

            # get the target
            data = pandas.read_csv(self.datapath[idx][1], header=0).values.astype(np.float32)
        
            # the points and target
            pos = torch.tensor(data[:,:3], dtype=torch.float)
            if self.use_normals:
                x = torch.tensor(data[:,3:], dtype=torch.float)
            else:
                x = torch.ones((pos.shape[0],3))
            y = int(self.cat.index(self.datapath[idx][0]))
            return Data(pos=pos, y=y, x=x)