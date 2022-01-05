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

import pandas as pd
from sklearn.neighbors import KDTree
import logging

class S3DIS(Dataset):

    def __init__(self,
                 root,
                 test_area=6,
                 train=True,
                 transform=None,
                 **kwargs):
        assert test_area >= 1 and test_area <= 6
        self.test_area = test_area
        super().__init__(root, transform, None)

        logging.info(f"S3DIS - training {train} - val area {test_area}")

        self.label_to_names = {0: 'ceiling',
                               1: 'floor',
                               2: 'wall',
                               3: 'beam',
                               4: 'column',
                               5: 'window',
                               6: 'door',
                               7: 'table',
                               8: 'chair',
                               9: 'sofa',
                               10: 'bookcase',
                               11: 'board',
                               12: 'clutter'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([])

        self.anno_paths = [
            "Area_1/conferenceRoom_1/Annotations",
            "Area_1/conferenceRoom_2/Annotations",
            "Area_1/copyRoom_1/Annotations",
            "Area_1/hallway_1/Annotations",
            "Area_1/hallway_2/Annotations",
            "Area_1/hallway_3/Annotations",
            "Area_1/hallway_4/Annotations",
            "Area_1/hallway_5/Annotations",
            "Area_1/hallway_6/Annotations",
            "Area_1/hallway_7/Annotations",
            "Area_1/hallway_8/Annotations",
            "Area_1/office_10/Annotations",
            "Area_1/office_11/Annotations",
            "Area_1/office_12/Annotations",
            "Area_1/office_13/Annotations",
            "Area_1/office_14/Annotations",
            "Area_1/office_15/Annotations",
            "Area_1/office_16/Annotations",
            "Area_1/office_17/Annotations",
            "Area_1/office_18/Annotations",
            "Area_1/office_19/Annotations",
            "Area_1/office_1/Annotations",
            "Area_1/office_20/Annotations",
            "Area_1/office_21/Annotations",
            "Area_1/office_22/Annotations",
            "Area_1/office_23/Annotations",
            "Area_1/office_24/Annotations",
            "Area_1/office_25/Annotations",
            "Area_1/office_26/Annotations",
            "Area_1/office_27/Annotations",
            "Area_1/office_28/Annotations",
            "Area_1/office_29/Annotations",
            "Area_1/office_2/Annotations",
            "Area_1/office_30/Annotations",
            "Area_1/office_31/Annotations",
            "Area_1/office_3/Annotations",
            "Area_1/office_4/Annotations",
            "Area_1/office_5/Annotations",
            "Area_1/office_6/Annotations",
            "Area_1/office_7/Annotations",
            "Area_1/office_8/Annotations",
            "Area_1/office_9/Annotations",
            "Area_1/pantry_1/Annotations",
            "Area_1/WC_1/Annotations",
            "Area_2/auditorium_1/Annotations",
            "Area_2/auditorium_2/Annotations",
            "Area_2/conferenceRoom_1/Annotations",
            "Area_2/hallway_10/Annotations",
            "Area_2/hallway_11/Annotations",
            "Area_2/hallway_12/Annotations",
            "Area_2/hallway_1/Annotations",
            "Area_2/hallway_2/Annotations",
            "Area_2/hallway_3/Annotations",
            "Area_2/hallway_4/Annotations",
            "Area_2/hallway_5/Annotations",
            "Area_2/hallway_6/Annotations",
            "Area_2/hallway_7/Annotations",
            "Area_2/hallway_8/Annotations",
            "Area_2/hallway_9/Annotations",
            "Area_2/office_10/Annotations",
            "Area_2/office_11/Annotations",
            "Area_2/office_12/Annotations",
            "Area_2/office_13/Annotations",
            "Area_2/office_14/Annotations",
            "Area_2/office_1/Annotations",
            "Area_2/office_2/Annotations",
            "Area_2/office_3/Annotations",
            "Area_2/office_4/Annotations",
            "Area_2/office_5/Annotations",
            "Area_2/office_6/Annotations",
            "Area_2/office_7/Annotations",
            "Area_2/office_8/Annotations",
            "Area_2/office_9/Annotations",
            "Area_2/storage_1/Annotations",
            "Area_2/storage_2/Annotations",
            "Area_2/storage_3/Annotations",
            "Area_2/storage_4/Annotations",
            "Area_2/storage_5/Annotations",
            "Area_2/storage_6/Annotations",
            "Area_2/storage_7/Annotations",
            "Area_2/storage_8/Annotations",
            "Area_2/storage_9/Annotations",
            "Area_2/WC_1/Annotations",
            "Area_2/WC_2/Annotations",
            "Area_3/conferenceRoom_1/Annotations",
            "Area_3/hallway_1/Annotations",
            "Area_3/hallway_2/Annotations",
            "Area_3/hallway_3/Annotations",
            "Area_3/hallway_4/Annotations",
            "Area_3/hallway_5/Annotations",
            "Area_3/hallway_6/Annotations",
            "Area_3/lounge_1/Annotations",
            "Area_3/lounge_2/Annotations",
            "Area_3/office_10/Annotations",
            "Area_3/office_1/Annotations",
            "Area_3/office_2/Annotations",
            "Area_3/office_3/Annotations",
            "Area_3/office_4/Annotations",
            "Area_3/office_5/Annotations",
            "Area_3/office_6/Annotations",
            "Area_3/office_7/Annotations",
            "Area_3/office_8/Annotations",
            "Area_3/office_9/Annotations",
            "Area_3/storage_1/Annotations",
            "Area_3/storage_2/Annotations",
            "Area_3/WC_1/Annotations",
            "Area_3/WC_2/Annotations",
            "Area_4/conferenceRoom_1/Annotations",
            "Area_4/conferenceRoom_2/Annotations",
            "Area_4/conferenceRoom_3/Annotations",
            "Area_4/hallway_10/Annotations",
            "Area_4/hallway_11/Annotations",
            "Area_4/hallway_12/Annotations",
            "Area_4/hallway_13/Annotations",
            "Area_4/hallway_14/Annotations",
            "Area_4/hallway_1/Annotations",
            "Area_4/hallway_2/Annotations",
            "Area_4/hallway_3/Annotations",
            "Area_4/hallway_4/Annotations",
            "Area_4/hallway_5/Annotations",
            "Area_4/hallway_6/Annotations",
            "Area_4/hallway_7/Annotations",
            "Area_4/hallway_8/Annotations",
            "Area_4/hallway_9/Annotations",
            "Area_4/lobby_1/Annotations",
            "Area_4/lobby_2/Annotations",
            "Area_4/office_10/Annotations",
            "Area_4/office_11/Annotations",
            "Area_4/office_12/Annotations",
            "Area_4/office_13/Annotations",
            "Area_4/office_14/Annotations",
            "Area_4/office_15/Annotations",
            "Area_4/office_16/Annotations",
            "Area_4/office_17/Annotations",
            "Area_4/office_18/Annotations",
            "Area_4/office_19/Annotations",
            "Area_4/office_1/Annotations",
            "Area_4/office_20/Annotations",
            "Area_4/office_21/Annotations",
            "Area_4/office_22/Annotations",
            "Area_4/office_2/Annotations",
            "Area_4/office_3/Annotations",
            "Area_4/office_4/Annotations",
            "Area_4/office_5/Annotations",
            "Area_4/office_6/Annotations",
            "Area_4/office_7/Annotations",
            "Area_4/office_8/Annotations",
            "Area_4/office_9/Annotations",
            "Area_4/storage_1/Annotations",
            "Area_4/storage_2/Annotations",
            "Area_4/storage_3/Annotations",
            "Area_4/storage_4/Annotations",
            "Area_4/WC_1/Annotations",
            "Area_4/WC_2/Annotations",
            "Area_4/WC_3/Annotations",
            "Area_4/WC_4/Annotations",
            "Area_5/conferenceRoom_1/Annotations",
            "Area_5/conferenceRoom_2/Annotations",
            "Area_5/conferenceRoom_3/Annotations",
            "Area_5/hallway_10/Annotations",
            "Area_5/hallway_11/Annotations",
            "Area_5/hallway_12/Annotations",
            "Area_5/hallway_13/Annotations",
            "Area_5/hallway_14/Annotations",
            "Area_5/hallway_15/Annotations",
            "Area_5/hallway_1/Annotations",
            "Area_5/hallway_2/Annotations",
            "Area_5/hallway_3/Annotations",
            "Area_5/hallway_4/Annotations",
            "Area_5/hallway_5/Annotations",
            "Area_5/hallway_6/Annotations",
            "Area_5/hallway_7/Annotations",
            "Area_5/hallway_8/Annotations",
            "Area_5/hallway_9/Annotations",
            "Area_5/lobby_1/Annotations",
            "Area_5/office_10/Annotations",
            "Area_5/office_11/Annotations",
            "Area_5/office_12/Annotations",
            "Area_5/office_13/Annotations",
            "Area_5/office_14/Annotations",
            "Area_5/office_15/Annotations",
            "Area_5/office_16/Annotations",
            "Area_5/office_17/Annotations",
            "Area_5/office_18/Annotations",
            "Area_5/office_19/Annotations",
            "Area_5/office_1/Annotations",
            "Area_5/office_20/Annotations",
            "Area_5/office_21/Annotations",
            "Area_5/office_22/Annotations",
            "Area_5/office_23/Annotations",
            "Area_5/office_24/Annotations",
            "Area_5/office_25/Annotations",
            "Area_5/office_26/Annotations",
            "Area_5/office_27/Annotations",
            "Area_5/office_28/Annotations",
            "Area_5/office_29/Annotations",
            "Area_5/office_2/Annotations",
            "Area_5/office_30/Annotations",
            "Area_5/office_31/Annotations",
            "Area_5/office_32/Annotations",
            "Area_5/office_33/Annotations",
            "Area_5/office_34/Annotations",
            "Area_5/office_35/Annotations",
            "Area_5/office_36/Annotations",
            "Area_5/office_37/Annotations",
            "Area_5/office_38/Annotations",
            "Area_5/office_39/Annotations",
            "Area_5/office_3/Annotations",
            "Area_5/office_40/Annotations",
            "Area_5/office_41/Annotations",
            "Area_5/office_42/Annotations",
            "Area_5/office_4/Annotations",
            "Area_5/office_5/Annotations",
            "Area_5/office_6/Annotations",
            "Area_5/office_7/Annotations",
            "Area_5/office_8/Annotations",
            "Area_5/office_9/Annotations",
            "Area_5/pantry_1/Annotations",
            "Area_5/storage_1/Annotations",
            "Area_5/storage_2/Annotations",
            "Area_5/storage_3/Annotations",
            "Area_5/storage_4/Annotations",
            "Area_5/WC_1/Annotations",
            "Area_5/WC_2/Annotations",
            "Area_6/conferenceRoom_1/Annotations",
            "Area_6/copyRoom_1/Annotations",
            "Area_6/hallway_1/Annotations",
            "Area_6/hallway_2/Annotations",
            "Area_6/hallway_3/Annotations",
            "Area_6/hallway_4/Annotations",
            "Area_6/hallway_5/Annotations",
            "Area_6/hallway_6/Annotations",
            "Area_6/lounge_1/Annotations",
            "Area_6/office_10/Annotations",
            "Area_6/office_11/Annotations",
            "Area_6/office_12/Annotations",
            "Area_6/office_13/Annotations",
            "Area_6/office_14/Annotations",
            "Area_6/office_15/Annotations",
            "Area_6/office_16/Annotations",
            "Area_6/office_17/Annotations",
            "Area_6/office_18/Annotations",
            "Area_6/office_19/Annotations",
            "Area_6/office_1/Annotations",
            "Area_6/office_20/Annotations",
            "Area_6/office_21/Annotations",
            "Area_6/office_22/Annotations",
            "Area_6/office_23/Annotations",
            "Area_6/office_24/Annotations",
            "Area_6/office_25/Annotations",
            "Area_6/office_26/Annotations",
            "Area_6/office_27/Annotations",
            "Area_6/office_28/Annotations",
            "Area_6/office_29/Annotations",
            "Area_6/office_2/Annotations",
            "Area_6/office_30/Annotations",
            "Area_6/office_31/Annotations",
            "Area_6/office_32/Annotations",
            "Area_6/office_33/Annotations",
            "Area_6/office_34/Annotations",
            "Area_6/office_35/Annotations",
            "Area_6/office_36/Annotations",
            "Area_6/office_37/Annotations",
            "Area_6/office_3/Annotations",
            "Area_6/office_4/Annotations",
            "Area_6/office_5/Annotations",
            "Area_6/office_6/Annotations",
            "Area_6/office_7/Annotations",
            "Area_6/office_8/Annotations",
            "Area_6/office_9/Annotations",
            "Area_6/openspace_1/Annotations",
            "Area_6/pantry_1/Annotations",]
        
        self.gt_class = [
            "ceiling",
            "floor",
            "wall",
            "beam",
            "column",
            "window",
            "door",
            "table",
            "chair",
            "sofa",
            "bookcase",
            "board",
            "clutter",]

        sub_grid_size = 0.040
        self.sub_grid_size = 0.040
        if (not os.path.exists(os.path.join(self.root, 'original_ply'))) or (not os.path.exists(os.path.join(self.root, f'input_{sub_grid_size:.3f}'))):
            # path does not exists

            anno_paths = [os.path.join(self.root, p) for p in self.anno_paths]
            gt_class2label = {cls: i for i, cls in enumerate(self.gt_class)}

            original_pc_folder = os.path.join(self.root, 'original_ply')
            sub_pc_folder = os.path.join(self.root, f'input_{sub_grid_size:.3f}')
            os.makedirs(original_pc_folder, exist_ok=True)
            os.makedirs(sub_pc_folder, exist_ok=True)
            out_format = '.ply'

            # Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
            for annotation_path in anno_paths:
                print(annotation_path)
                elements = str(annotation_path).split('/')
                out_file_name = elements[-3] + '_' + elements[-2] + out_format

                save_path = os.path.join(original_pc_folder, out_file_name)

                # convert_pc2ply(annotation_path, save_path)

                data_list = []

                for f in glob.glob(os.path.join(annotation_path, '*.txt')):
                    class_name = os.path.basename(f).split('_')[0]
                    if class_name not in self.gt_class:  # note: in some room there is 'staris' class..
                        class_name = 'clutter'
                    pc = pd.read_csv(f, header=None, delim_whitespace=True).values
                    labels = np.ones((pc.shape[0], 1)) * gt_class2label[class_name]
                    data_list.append(np.concatenate([pc, labels], 1))  # Nx7

                pc_label = np.concatenate(data_list, 0)
                xyz_min = np.amin(pc_label, axis=0)[0:3]
                pc_label[:, 0:3] -= xyz_min

                xyz = pc_label[:, :3].astype(np.float32)
                colors = pc_label[:, 3:6].astype(np.uint8)
                labels = pc_label[:, 6].astype(np.uint8)
                write_ply(save_path, (xyz, colors, labels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

                # save sub_cloud and KDTree file
                sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(xyz, colors, labels, sub_grid_size)
                sub_colors = sub_colors / 255.0
                sub_ply_file = os.path.join(sub_pc_folder, save_path.split('/')[-1][:-4] + '.ply')
                write_ply(sub_ply_file, [sub_xyz, sub_colors, sub_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

                search_tree = KDTree(sub_xyz)
                kd_tree_file = os.path.join(sub_pc_folder, str(save_path.split('/')[-1][:-4]) + '_KDTree.pkl')
                with open(kd_tree_file, 'wb') as f:
                    pickle.dump(search_tree, f)

                proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
                proj_idx = proj_idx.astype(np.int32)
                proj_save = os.path.join(sub_pc_folder, str(save_path.split('/')[-1][:-4]) + '_proj.pkl')
                with open(proj_save, 'wb') as f:
                    pickle.dump([proj_idx, labels], f)

        filenames = glob.glob(os.path.join(self.root, 'original_ply', '*.ply'))

        val_area_name = 'Area_' + str(test_area)
        self.all_files = []
        for filename in filenames:
            if train and val_area_name not in filename:
                self.all_files.append(filename)
            elif (not train) and (val_area_name in filename):
                self.all_files.append(filename)


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
        return len(self.all_files)

    def get(self, idx):
        """Get item."""

        tree_path = os.path.join(self.root, 'input_{:.3f}'.format(self.sub_grid_size))
        file_path = self.all_files[idx]
        cloud_name = file_path.split('/')[-1][:-4]
        sub_ply_file = os.path.join(tree_path, '{:s}.ply'.format(cloud_name))
        data = read_ply(sub_ply_file)
        sub_points = np.vstack((data['x'], data['y'], data['z'])).T
        sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
        sub_labels = data['class']

        x = torch.tensor(sub_colors, dtype=torch.float)
        pos = torch.tensor(sub_points, dtype=torch.float)
        y = torch.tensor(sub_labels, dtype=torch.long)

        data = Data(pos=pos, y=y, x=x)

        return data


class S3DIS_TrainVal_NumIter(Dataset):

    def __init__(self,
                 root,
                 test_area=6,
                 train=True,
                 transform=None,
                 num_iter=1000,
                 **kwargs):
        assert test_area >= 1 and test_area <= 6
        self.test_area = test_area
        super().__init__(root, transform, None)
        self.num_iter = num_iter

        logging.info(f"S3DIS - training {train} - val area {test_area}")

        self.label_to_names = {0: 'ceiling',
                               1: 'floor',
                               2: 'wall',
                               3: 'beam',
                               4: 'column',
                               5: 'window',
                               6: 'door',
                               7: 'table',
                               8: 'chair',
                               9: 'sofa',
                               10: 'bookcase',
                               11: 'board',
                               12: 'clutter'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([])

        self.anno_paths = [
            "Area_1/conferenceRoom_1/Annotations",
            "Area_1/conferenceRoom_2/Annotations",
            "Area_1/copyRoom_1/Annotations",
            "Area_1/hallway_1/Annotations",
            "Area_1/hallway_2/Annotations",
            "Area_1/hallway_3/Annotations",
            "Area_1/hallway_4/Annotations",
            "Area_1/hallway_5/Annotations",
            "Area_1/hallway_6/Annotations",
            "Area_1/hallway_7/Annotations",
            "Area_1/hallway_8/Annotations",
            "Area_1/office_10/Annotations",
            "Area_1/office_11/Annotations",
            "Area_1/office_12/Annotations",
            "Area_1/office_13/Annotations",
            "Area_1/office_14/Annotations",
            "Area_1/office_15/Annotations",
            "Area_1/office_16/Annotations",
            "Area_1/office_17/Annotations",
            "Area_1/office_18/Annotations",
            "Area_1/office_19/Annotations",
            "Area_1/office_1/Annotations",
            "Area_1/office_20/Annotations",
            "Area_1/office_21/Annotations",
            "Area_1/office_22/Annotations",
            "Area_1/office_23/Annotations",
            "Area_1/office_24/Annotations",
            "Area_1/office_25/Annotations",
            "Area_1/office_26/Annotations",
            "Area_1/office_27/Annotations",
            "Area_1/office_28/Annotations",
            "Area_1/office_29/Annotations",
            "Area_1/office_2/Annotations",
            "Area_1/office_30/Annotations",
            "Area_1/office_31/Annotations",
            "Area_1/office_3/Annotations",
            "Area_1/office_4/Annotations",
            "Area_1/office_5/Annotations",
            "Area_1/office_6/Annotations",
            "Area_1/office_7/Annotations",
            "Area_1/office_8/Annotations",
            "Area_1/office_9/Annotations",
            "Area_1/pantry_1/Annotations",
            "Area_1/WC_1/Annotations",
            "Area_2/auditorium_1/Annotations",
            "Area_2/auditorium_2/Annotations",
            "Area_2/conferenceRoom_1/Annotations",
            "Area_2/hallway_10/Annotations",
            "Area_2/hallway_11/Annotations",
            "Area_2/hallway_12/Annotations",
            "Area_2/hallway_1/Annotations",
            "Area_2/hallway_2/Annotations",
            "Area_2/hallway_3/Annotations",
            "Area_2/hallway_4/Annotations",
            "Area_2/hallway_5/Annotations",
            "Area_2/hallway_6/Annotations",
            "Area_2/hallway_7/Annotations",
            "Area_2/hallway_8/Annotations",
            "Area_2/hallway_9/Annotations",
            "Area_2/office_10/Annotations",
            "Area_2/office_11/Annotations",
            "Area_2/office_12/Annotations",
            "Area_2/office_13/Annotations",
            "Area_2/office_14/Annotations",
            "Area_2/office_1/Annotations",
            "Area_2/office_2/Annotations",
            "Area_2/office_3/Annotations",
            "Area_2/office_4/Annotations",
            "Area_2/office_5/Annotations",
            "Area_2/office_6/Annotations",
            "Area_2/office_7/Annotations",
            "Area_2/office_8/Annotations",
            "Area_2/office_9/Annotations",
            "Area_2/storage_1/Annotations",
            "Area_2/storage_2/Annotations",
            "Area_2/storage_3/Annotations",
            "Area_2/storage_4/Annotations",
            "Area_2/storage_5/Annotations",
            "Area_2/storage_6/Annotations",
            "Area_2/storage_7/Annotations",
            "Area_2/storage_8/Annotations",
            "Area_2/storage_9/Annotations",
            "Area_2/WC_1/Annotations",
            "Area_2/WC_2/Annotations",
            "Area_3/conferenceRoom_1/Annotations",
            "Area_3/hallway_1/Annotations",
            "Area_3/hallway_2/Annotations",
            "Area_3/hallway_3/Annotations",
            "Area_3/hallway_4/Annotations",
            "Area_3/hallway_5/Annotations",
            "Area_3/hallway_6/Annotations",
            "Area_3/lounge_1/Annotations",
            "Area_3/lounge_2/Annotations",
            "Area_3/office_10/Annotations",
            "Area_3/office_1/Annotations",
            "Area_3/office_2/Annotations",
            "Area_3/office_3/Annotations",
            "Area_3/office_4/Annotations",
            "Area_3/office_5/Annotations",
            "Area_3/office_6/Annotations",
            "Area_3/office_7/Annotations",
            "Area_3/office_8/Annotations",
            "Area_3/office_9/Annotations",
            "Area_3/storage_1/Annotations",
            "Area_3/storage_2/Annotations",
            "Area_3/WC_1/Annotations",
            "Area_3/WC_2/Annotations",
            "Area_4/conferenceRoom_1/Annotations",
            "Area_4/conferenceRoom_2/Annotations",
            "Area_4/conferenceRoom_3/Annotations",
            "Area_4/hallway_10/Annotations",
            "Area_4/hallway_11/Annotations",
            "Area_4/hallway_12/Annotations",
            "Area_4/hallway_13/Annotations",
            "Area_4/hallway_14/Annotations",
            "Area_4/hallway_1/Annotations",
            "Area_4/hallway_2/Annotations",
            "Area_4/hallway_3/Annotations",
            "Area_4/hallway_4/Annotations",
            "Area_4/hallway_5/Annotations",
            "Area_4/hallway_6/Annotations",
            "Area_4/hallway_7/Annotations",
            "Area_4/hallway_8/Annotations",
            "Area_4/hallway_9/Annotations",
            "Area_4/lobby_1/Annotations",
            "Area_4/lobby_2/Annotations",
            "Area_4/office_10/Annotations",
            "Area_4/office_11/Annotations",
            "Area_4/office_12/Annotations",
            "Area_4/office_13/Annotations",
            "Area_4/office_14/Annotations",
            "Area_4/office_15/Annotations",
            "Area_4/office_16/Annotations",
            "Area_4/office_17/Annotations",
            "Area_4/office_18/Annotations",
            "Area_4/office_19/Annotations",
            "Area_4/office_1/Annotations",
            "Area_4/office_20/Annotations",
            "Area_4/office_21/Annotations",
            "Area_4/office_22/Annotations",
            "Area_4/office_2/Annotations",
            "Area_4/office_3/Annotations",
            "Area_4/office_4/Annotations",
            "Area_4/office_5/Annotations",
            "Area_4/office_6/Annotations",
            "Area_4/office_7/Annotations",
            "Area_4/office_8/Annotations",
            "Area_4/office_9/Annotations",
            "Area_4/storage_1/Annotations",
            "Area_4/storage_2/Annotations",
            "Area_4/storage_3/Annotations",
            "Area_4/storage_4/Annotations",
            "Area_4/WC_1/Annotations",
            "Area_4/WC_2/Annotations",
            "Area_4/WC_3/Annotations",
            "Area_4/WC_4/Annotations",
            "Area_5/conferenceRoom_1/Annotations",
            "Area_5/conferenceRoom_2/Annotations",
            "Area_5/conferenceRoom_3/Annotations",
            "Area_5/hallway_10/Annotations",
            "Area_5/hallway_11/Annotations",
            "Area_5/hallway_12/Annotations",
            "Area_5/hallway_13/Annotations",
            "Area_5/hallway_14/Annotations",
            "Area_5/hallway_15/Annotations",
            "Area_5/hallway_1/Annotations",
            "Area_5/hallway_2/Annotations",
            "Area_5/hallway_3/Annotations",
            "Area_5/hallway_4/Annotations",
            "Area_5/hallway_5/Annotations",
            "Area_5/hallway_6/Annotations",
            "Area_5/hallway_7/Annotations",
            "Area_5/hallway_8/Annotations",
            "Area_5/hallway_9/Annotations",
            "Area_5/lobby_1/Annotations",
            "Area_5/office_10/Annotations",
            "Area_5/office_11/Annotations",
            "Area_5/office_12/Annotations",
            "Area_5/office_13/Annotations",
            "Area_5/office_14/Annotations",
            "Area_5/office_15/Annotations",
            "Area_5/office_16/Annotations",
            "Area_5/office_17/Annotations",
            "Area_5/office_18/Annotations",
            "Area_5/office_19/Annotations",
            "Area_5/office_1/Annotations",
            "Area_5/office_20/Annotations",
            "Area_5/office_21/Annotations",
            "Area_5/office_22/Annotations",
            "Area_5/office_23/Annotations",
            "Area_5/office_24/Annotations",
            "Area_5/office_25/Annotations",
            "Area_5/office_26/Annotations",
            "Area_5/office_27/Annotations",
            "Area_5/office_28/Annotations",
            "Area_5/office_29/Annotations",
            "Area_5/office_2/Annotations",
            "Area_5/office_30/Annotations",
            "Area_5/office_31/Annotations",
            "Area_5/office_32/Annotations",
            "Area_5/office_33/Annotations",
            "Area_5/office_34/Annotations",
            "Area_5/office_35/Annotations",
            "Area_5/office_36/Annotations",
            "Area_5/office_37/Annotations",
            "Area_5/office_38/Annotations",
            "Area_5/office_39/Annotations",
            "Area_5/office_3/Annotations",
            "Area_5/office_40/Annotations",
            "Area_5/office_41/Annotations",
            "Area_5/office_42/Annotations",
            "Area_5/office_4/Annotations",
            "Area_5/office_5/Annotations",
            "Area_5/office_6/Annotations",
            "Area_5/office_7/Annotations",
            "Area_5/office_8/Annotations",
            "Area_5/office_9/Annotations",
            "Area_5/pantry_1/Annotations",
            "Area_5/storage_1/Annotations",
            "Area_5/storage_2/Annotations",
            "Area_5/storage_3/Annotations",
            "Area_5/storage_4/Annotations",
            "Area_5/WC_1/Annotations",
            "Area_5/WC_2/Annotations",
            "Area_6/conferenceRoom_1/Annotations",
            "Area_6/copyRoom_1/Annotations",
            "Area_6/hallway_1/Annotations",
            "Area_6/hallway_2/Annotations",
            "Area_6/hallway_3/Annotations",
            "Area_6/hallway_4/Annotations",
            "Area_6/hallway_5/Annotations",
            "Area_6/hallway_6/Annotations",
            "Area_6/lounge_1/Annotations",
            "Area_6/office_10/Annotations",
            "Area_6/office_11/Annotations",
            "Area_6/office_12/Annotations",
            "Area_6/office_13/Annotations",
            "Area_6/office_14/Annotations",
            "Area_6/office_15/Annotations",
            "Area_6/office_16/Annotations",
            "Area_6/office_17/Annotations",
            "Area_6/office_18/Annotations",
            "Area_6/office_19/Annotations",
            "Area_6/office_1/Annotations",
            "Area_6/office_20/Annotations",
            "Area_6/office_21/Annotations",
            "Area_6/office_22/Annotations",
            "Area_6/office_23/Annotations",
            "Area_6/office_24/Annotations",
            "Area_6/office_25/Annotations",
            "Area_6/office_26/Annotations",
            "Area_6/office_27/Annotations",
            "Area_6/office_28/Annotations",
            "Area_6/office_29/Annotations",
            "Area_6/office_2/Annotations",
            "Area_6/office_30/Annotations",
            "Area_6/office_31/Annotations",
            "Area_6/office_32/Annotations",
            "Area_6/office_33/Annotations",
            "Area_6/office_34/Annotations",
            "Area_6/office_35/Annotations",
            "Area_6/office_36/Annotations",
            "Area_6/office_37/Annotations",
            "Area_6/office_3/Annotations",
            "Area_6/office_4/Annotations",
            "Area_6/office_5/Annotations",
            "Area_6/office_6/Annotations",
            "Area_6/office_7/Annotations",
            "Area_6/office_8/Annotations",
            "Area_6/office_9/Annotations",
            "Area_6/openspace_1/Annotations",
            "Area_6/pantry_1/Annotations",]
        
        self.gt_class = [
            "ceiling",
            "floor",
            "wall",
            "beam",
            "column",
            "window",
            "door",
            "table",
            "chair",
            "sofa",
            "bookcase",
            "board",
            "clutter",]

        sub_grid_size = 0.040
        self.sub_grid_size = 0.040
        if (not os.path.exists(os.path.join(self.root, 'original_ply'))) or (not os.path.exists(os.path.join(self.root, f'input_{sub_grid_size:.3f}'))):
            # path does not exists

            anno_paths = [os.path.join(self.root, p) for p in self.anno_paths]
            gt_class2label = {cls: i for i, cls in enumerate(self.gt_class)}

            original_pc_folder = os.path.join(self.root, 'original_ply')
            sub_pc_folder = os.path.join(self.root, f'input_{sub_grid_size:.3f}')
            os.makedirs(original_pc_folder, exist_ok=True)
            os.makedirs(sub_pc_folder, exist_ok=True)
            out_format = '.ply'

            # Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
            for annotation_path in anno_paths:
                print(annotation_path)
                elements = str(annotation_path).split('/')
                out_file_name = elements[-3] + '_' + elements[-2] + out_format

                save_path = os.path.join(original_pc_folder, out_file_name)

                # convert_pc2ply(annotation_path, save_path)

                data_list = []

                for f in glob.glob(os.path.join(annotation_path, '*.txt')):
                    class_name = os.path.basename(f).split('_')[0]
                    if class_name not in self.gt_class:  # note: in some room there is 'staris' class..
                        class_name = 'clutter'
                    pc = pd.read_csv(f, header=None, delim_whitespace=True).values
                    labels = np.ones((pc.shape[0], 1)) * gt_class2label[class_name]
                    data_list.append(np.concatenate([pc, labels], 1))  # Nx7

                pc_label = np.concatenate(data_list, 0)
                xyz_min = np.amin(pc_label, axis=0)[0:3]
                pc_label[:, 0:3] -= xyz_min

                xyz = pc_label[:, :3].astype(np.float32)
                colors = pc_label[:, 3:6].astype(np.uint8)
                labels = pc_label[:, 6].astype(np.uint8)
                write_ply(save_path, (xyz, colors, labels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

                # save sub_cloud and KDTree file
                sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(xyz, colors, labels, sub_grid_size)
                sub_colors = sub_colors / 255.0
                sub_ply_file = os.path.join(sub_pc_folder, save_path.split('/')[-1][:-4] + '.ply')
                write_ply(sub_ply_file, [sub_xyz, sub_colors, sub_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

                search_tree = KDTree(sub_xyz)
                kd_tree_file = os.path.join(sub_pc_folder, str(save_path.split('/')[-1][:-4]) + '_KDTree.pkl')
                with open(kd_tree_file, 'wb') as f:
                    pickle.dump(search_tree, f)

                proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
                proj_idx = proj_idx.astype(np.int32)
                proj_save = os.path.join(sub_pc_folder, str(save_path.split('/')[-1][:-4]) + '_proj.pkl')
                with open(proj_save, 'wb') as f:
                    pickle.dump([proj_idx, labels], f)

        filenames = glob.glob(os.path.join(self.root, 'original_ply', '*.ply'))

        val_area_name = 'Area_' + str(test_area)
        self.all_files = []
        for filename in filenames:
            if train and val_area_name not in filename:
                self.all_files.append(filename)
            elif (not train) and (val_area_name in filename):
                self.all_files.append(filename)

        # computing the picking probability
        self.prob_all_files = []
        for idx in range(len(self.all_files)):
            tree_path = os.path.join(self.root, 'input_{:.3f}'.format(self.sub_grid_size))
            file_path = self.all_files[idx]
            cloud_name = file_path.split('/')[-1][:-4]
            sub_ply_file = os.path.join(tree_path, '{:s}.ply'.format(cloud_name))
            data = read_ply(sub_ply_file)
            num_points_in_file = data['x'].shape[0]
            self.prob_all_files.append(num_points_in_file)
        self.prob_all_files = np.array(self.prob_all_files, dtype=np.float64)
        self.prob_all_files /= self.prob_all_files.sum()
        self.prob_all_files = np.cumsum(self.prob_all_files)

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

    def _process(self):
        pass

    def process(self):
        pass

    def len(self):
        return self.num_iter

    def get(self, idx):
        """Get item."""

        # pick a random number
        # prob = torch.rand((1,)).item()
        # idx = 0
        # for id_file, prob_file in enumerate(self.prob_all_files):
        #     if prob < prob_file:
        #         idx = id_file
        #         break

        idx = idx % len(self.all_files)

        # tree_path = os.path.join(self.root, 'input_{:.3f}'.format(self.sub_grid_size))
        # file_path = self.all_files[idx]
        # cloud_name = file_path.split('/')[-1][:-4]
        # sub_ply_file = os.path.join(tree_path, '{:s}.ply'.format(cloud_name))
        # data = read_ply(sub_ply_file)
        # sub_points = np.vstack((data['x'], data['y'], data['z'])).T
        # sub_points = sub_points.astype(np.float32)
        # sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
        # sub_labels = data['class']

        tree_path = os.path.join(self.root, 'original_ply'.format(self.sub_grid_size))
        file_path = self.all_files[idx]
        cloud_name = file_path.split('/')[-1][:-4]
        sub_ply_file = os.path.join(tree_path, '{:s}.ply'.format(cloud_name))
        data = read_ply(sub_ply_file)
        sub_points = np.vstack((data['x'], data['y'], data['z'])).T
        sub_points = sub_points.astype(np.float32)
        sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
        sub_colors = sub_colors.astype(np.float32)/255
        sub_labels = data['class']



        x = torch.tensor(sub_colors, dtype=torch.float)
        pos = torch.tensor(sub_points, dtype=torch.float)
        y = torch.tensor(sub_labels, dtype=torch.long)

        data = Data(pos=pos, y=y, x=x)

        return data