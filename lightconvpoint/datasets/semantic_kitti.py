import os
import numpy as np
import glob
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import pickle
from .helper_ply import write_ply, read_ply
from .helper_tool import DataProcessing as DP
import importlib
if importlib.util.find_spec("valeodata") is not None:
    valeodata_exists = True
    import valeodata
else:
    valeodata_exists = False

from sklearn.neighbors import KDTree
import logging
from pathlib import Path
import yaml

class SemanticKITTI(Dataset):

    def __init__(self,
                 root,
                 split="train",
                 transform=None,
                 pre_grid_sampling=None,
                 **kwargs):
        self.split = split
        self.n_frames = 1
        self.pre_grid_sampling = pre_grid_sampling
        super().__init__(root, transform, None)
        logging.info(f"SemanticKITTI - split {split}")


        

        for i in range(len(self.all_files)):
            fname = str(self.all_files[i]).split("/")[-3:]
            fname = os.path.join(fname[0], fname[1], fname[2])
            self.all_files[i] = fname

        # Read labels
        if self.n_frames == 1:
            config_file = os.path.join(self.root, 'semantic-kitti.yaml')
        elif self.n_frames > 1:
            config_file = os.path.join(self.root, 'semantic-kitti-all.yaml')
        else:
            raise ValueError('number of frames has to be >= 1')

        with open(config_file, 'r') as stream:
            doc = yaml.safe_load(stream)
            all_labels = doc['labels']
            learning_map_inv = doc['learning_map_inv']
            learning_map = doc['learning_map']
            self.learning_map = np.zeros((np.max([k for k in learning_map.keys()]) + 1), dtype=np.int32)
            for k, v in learning_map.items():
                self.learning_map[k] = v

            self.learning_map_inv = np.zeros((np.max([k for k in learning_map_inv.keys()]) + 1), dtype=np.int32)
            for k, v in learning_map_inv.items():
                self.learning_map_inv[k] = v

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def _download(self): # override _download to remove makedirs
        self.download()

    def download(self):

        if valeodata_exists:
            if self.pre_grid_sampling is not None:
                try:
                    self.root = valeodata.download(os.path.join(self.root, f"processed_{self.pre_grid_sampling}_{self.n_frames}"))
                    self.is_preprocessed=True
                except FileExistsError:
                    logging.warning(f"preprocessed dataset processed_{self.pre_grid_sampling}_{self.n_frames} does not exist, downloading full dataset")
                    self.root = valeodata.download(self.root)
                    self.is_preprocessed=False
            else:
                self.root = valeodata.download(self.root)
                self.is_preprocessed=True # no need for preprocessings

        else:
            if self.pre_grid_sampling is not None:
                if f"processed_{self.pre_grid_sampling}_{self.n_frames}" in self.root:
                    self.is_preprocessed = True
                else:
                    self.is_preprocessed = False
        
        logging.info(f"Dataset at {self.root}, preprocessed {self.is_preprocessed}")

    def _process(self):

        # Get a list of sequences
        if self.split in ["training", "train"]:
            self.sequences = ['{:02d}'.format(i) for i in range(11) if i != 8]
        elif self.split in ["validation", "val"]:
            self.sequences = ['{:02d}'.format(i) for i in range(11) if i == 8]
        elif self.split in ["trainVal", "train_val"]:
            self.sequences = ['{:02d}'.format(i) for i in range(11)]
        elif self.split in ["test"]:
            self.sequences = ['{:02d}'.format(i) for i in range(11, 22)]
        else:
            raise ValueError('Unknown set for SemanticKitti data: ', self.set)

        if not self.is_preprocessed: # requiring a pre-processing
            self.process()

        # get the filenames
        self.all_files = []
        for sequence in self.sequences:
            if self.pre_grid_sampling is not None:
                self.all_files += [path for path in Path(os.path.join(self.root, sequence, "velodyne")).rglob('*.ply')]
            else:
                self.all_files += [path for path in Path(os.path.join(self.root, "dataset", "sequences", sequence, "velodyne")).rglob('*.bin')]


    def process(self):
        raise NotImplementedError

    def len(self):
        return len(self.all_files)

    def get(self, idx):
        """Get item."""

        if self.pre_grid_sampling:
            fname_points = self.all_files[idx]
            data = read_ply(os.path.join(self.root, fname_points))
            pos = np.vstack((data['x'], data['y'], data['z'])).T
            x = np.ones((pos.shape[0], 1), dtype=np.float32)
            y = data['class']

        else:
            
            # get the filenamess
            fname_points = self.all_files[idx]
            fname_labels = str(fname_points).replace("velodyne", "labels").replace(".bin", ".label")

            # load the data
            pos = np.fromfile(os.path.join(self.root, "dataset", "sequences",fname_points), dtype=np.float32).reshape((-1, 4))
            pos = pos[:,:3]
            y = np.fromfile(os.path.join(self.root, "dataset", "sequences",fname_labels), dtype=np.int32)
            y = y & 0xFFFF  # semantic label in lower half
            y = self.learning_map[y]

            # remove unlabeled data
            mask = (y>0)
            pos = pos[mask]
            y = y[mask]
            x = np.ones((pos.shape[0], 1), dtype=np.float32)

        pos = torch.tensor(pos, dtype=torch.float)
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)

        data = Data(pos=pos, y=y, x=x)
        return data


        # if self.pre_grid_sampling is not None:

        #     dir_processed = os.path.join(self.root, f"processed_{self.pre_grid_sampling}_{self.n_frames}")
        #     fname_processed_points = os.path.join(dir_processed, str(fname_points).replace(".bin",".ply"))
        #     fname_processed_proj = os.path.join(dir_processed, str(fname_points).replace(".bin","_proj.pkl"))

        #     if not os.path.isfile(fname_processed_points):
        #         pos = np.fromfile(os.path.join(self.root, "dataset", "sequences",fname_points), dtype=np.float32).reshape((-1, 4))
        #         x = pos[:,3:]
        #         pos = pos[:,:3]
        #         y = np.fromfile(os.path.join(self.root, "dataset", "sequences",fname_labels), dtype=np.int32)
        #         y = y & 0xFFFF  # semantic label in lower half
        #         y = self.learning_map[y]

        #         pos = pos.astype(np.float32)
        #         x = np.repeat((x*255).astype(np.uint8), 3, axis=1)
        #         y = y.astype(np.uint8)

        #         mask = (y>0)
        #         pos = pos[mask]
        #         x = x[mask]
        #         y = y[mask]

        #         sub_pos, sub_x, sub_y = DP.grid_sub_sampling(pos, x, y, self.pre_grid_sampling)

        #         os.makedirs(os.path.dirname(fname_processed_points), exist_ok=True)
        #         # write_ply(fname_processed_points, (sub_pos, sub_x, sub_y), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])
        #         write_ply(fname_processed_points, (sub_pos, sub_y), ['x', 'y', 'z', 'class'])

        #         search_tree = KDTree(sub_pos)
        #         proj_idx = np.squeeze(search_tree.query(pos, return_distance=False))
        #         proj_idx = proj_idx.astype(np.int32)
        #         with open(fname_processed_proj, 'wb') as f:
        #             pickle.dump([proj_idx, y], f)

        #     data = read_ply(fname_processed_points)
        #     pos = np.vstack((data['x'], data['y'], data['z'])).T
        #     # x = np.expand_dims(data['red'].astype(np.float32)/255, axis=1)
        #     x = np.ones((pos.shape[0], 1), dtype=np.float32)
        #     y = data['class']


        # else:
        #     pos = np.fromfile(os.path.join(self.root, "dataset", "sequences",fname_points), dtype=np.float32).reshape((-1, 4))
        #     pos = pos[:,:3]
        #     y = np.fromfile(os.path.join(self.root, "dataset", "sequences",fname_labels), dtype=np.int32)
        #     y = y & 0xFFFF  # semantic label in lower half
        #     y = self.learning_map[y]

        #     mask = (y>0)
        #     pos = pos[mask]
        #     y = y[mask]

        #     x = np.ones((pos.shape[0], 1), dtype=np.float32)

        # pos = torch.tensor(pos, dtype=torch.float)
        # x = torch.tensor(x, dtype=torch.float)
        # y = torch.tensor(y, dtype=torch.long)

        # data = Data(pos=pos, y=y, x=x)

        # return data