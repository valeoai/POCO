from sklearn.neighbors import KDTree
#from os.path import join, exists, dirname, abspath
import numpy as np
import pandas as pd
import os, sys, glob, pickle
import time
import torch
from .helper_ply import write_ply, read_ply
from .helper_tool import DataProcessing as DP
from lightconvpoint.nn import with_indices_computation_rotation


class S3DIS:

    def __init__ (self,
                dataset_dir,
                config,
                split="training",
                verbose=False,
                network_function=None,
                transformations_data=None,
                transformations_points=None,
                transformations_features=None):

        self.split = split
        self.dataset_dir = dataset_dir
        self.cfg = config
        self.verbose = verbose
        self.t_data = transformations_data
        self.t_points = transformations_points
        self.t_features = transformations_features

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
        self.val_split = 'Area_' + str(config['dataset_val_area'])

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

        sub_grid_size = config['dataset_sub_grid_size']
        if (not os.path.exists(os.path.join(self.dataset_dir, 'original_ply'))) or (not os.path.exists(os.path.join(self.dataset_dir, f'input_{sub_grid_size:.3f}'))):
            # path does not exists

            anno_paths = [os.path.join(self.dataset_dir, p) for p in self.anno_paths]
            gt_class2label = {cls: i for i, cls in enumerate(self.gt_class)}

            original_pc_folder = os.path.join(self.dataset_dir, 'original_ply')
            sub_pc_folder = os.path.join(self.dataset_dir, f'input_{sub_grid_size:.3f}')
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


        self.all_files = glob.glob(os.path.join(self.dataset_dir, 'original_ply', '*.ply'))

        # Initiate containers
        self.input_trees = []
        self.input_colors = []
        self.input_labels = []
        self.input_names = []
        self.val_proj = []
        self.val_labels = []

        self.load_sub_sampled_clouds(sub_grid_size)
        
        self.possibility = None
        self.min_possibility = None

        if network_function is not None:
            self.net = network_function()
        else:
            self.net = None


    def load_sub_sampled_clouds(self, sub_grid_size):
        tree_path = os.path.join(self.dataset_dir, 'input_{:.3f}'.format(sub_grid_size))
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]
            if self.val_split in cloud_name:
                cloud_split = 'validation'
            else:
                cloud_split = 'training'

            if cloud_split != self.split:
                continue

            # Name of the input files
            kd_tree_file = os.path.join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = os.path.join(tree_path, '{:s}.ply'.format(cloud_name))

            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            sub_labels = data['class']

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees += [search_tree]
            self.input_colors += [sub_colors]
            self.input_labels += [sub_labels]
            self.input_names += [cloud_name]

            size = sub_colors.shape[0] * 4 * 7
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.split('/')[-1], size * 1e-6, time.time() - t0))


        # Get validation and test reprojected indices
        if self.split=='validation':
            print('\nPreparing reprojected indices for testing')
            for i, file_path in enumerate(self.all_files):
                t0 = time.time()
                cloud_name = file_path.split('/')[-1][:-4]

                # Validation projection and labels
                if self.val_split in cloud_name:
                    proj_file = os.path.join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                    with open(proj_file, 'rb') as f:
                        proj_idx, labels = pickle.load(f)
                    self.val_proj += [proj_idx]
                    self.val_labels += [labels]
                    print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))


    def __len__(self):
        if self.split == "training":
            return self.cfg['epoch_size']
        elif self.split == "validation":
            return self.cfg['val_size']



    def init_possibility(self):
        
        if self.verbose:
            print("Init possibility...", end="", flush=True) 

        self.possibility = []
        self.min_possibility = []
        # Random initialize
        for _, tree in enumerate(self.input_colors):
            self.possibility += [torch.rand(tree.data.shape[0]).numpy() * 1e-3]
            self.min_possibility += [float(np.min(self.possibility[-1]))]
        
        if self.verbose:
            print("Done", end="", flush=True) 

    @with_indices_computation_rotation
    def __getitem__(self, index):

        if self.possibility is None:
            self.init_possibility()

        # Choose the cloud with the lowest probability
        cloud_idx = int(np.argmin(self.min_possibility))

        # choose the point with the minimum of possibility in the cloud as query point
        point_ind = np.argmin(self.possibility[cloud_idx])

        # Get all points within the cloud from tree structure
        points = np.array(self.input_trees[cloud_idx].data, copy=False)

        # Center point of input region
        center_point = points[point_ind, :].reshape(1, -1)

        # Add noise to the center point
        noise = np.random.normal(scale=self.cfg['noise_init'] / 10, size=center_point.shape)
        pick_point = center_point + noise.astype(center_point.dtype)

        # Check if the number of points in the selected cloud is less than the predefined num_points
        if len(points) < self.cfg['num_points']:
            # Query all points within the cloud
            queried_idx = self.input_trees[cloud_idx].query(pick_point, k=len(points))[1][0]
            additional = torch.randint(low=0, high=len(points), size=[self.cfg['num_points']-len(points)]).numpy()
            queried_idx = np.concatenate([queried_idx, additional], axis=0)
        else:
            # Query the predefined number of points
            queried_idx = self.input_trees[cloud_idx].query(pick_point, k=self.cfg["num_points"])[1][0]

        # Shuffle index
        queried_idx = DP.shuffle_idx(queried_idx)
        # Get corresponding points and colors based on the index
        queried_pc_xyz = points[queried_idx]
        queried_pc_xyz = queried_pc_xyz - pick_point
        queried_pc_colors = self.input_colors[cloud_idx][queried_idx]
        queried_pc_labels = self.input_labels[cloud_idx][queried_idx]

        # Update the possibility of the selected points
        dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
        delta = np.square(1 - dists / np.max(dists))
        self.possibility[cloud_idx][queried_idx] += delta
        self.min_possibility[cloud_idx] = float(np.min(self.possibility[cloud_idx]))

        # up_sampled with replacement
        if len(points) < self.cfg["num_points"]:
            queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, self.cfg['num_points'])


        queried_pc_xyz = torch.tensor(queried_pc_xyz, dtype=torch.float)
        queried_pc_colors = torch.tensor(queried_pc_colors, dtype=torch.float)
        queried_pc_labels = torch.tensor(queried_pc_labels, dtype=torch.long)
        queried_idx = torch.tensor(queried_idx, dtype=torch.long)

        return {
            "pts": queried_pc_xyz.transpose(0,1),
            "features": queried_pc_colors.transpose(0,1),
            "targets": queried_pc_labels,
            "pts_idx": queried_idx,
            "cloud_idx": cloud_idx 
        }