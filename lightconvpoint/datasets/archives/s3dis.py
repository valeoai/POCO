import torch
import numpy as np
import os
import random
from tqdm import *
from lightconvpoint.nn import with_indices_computation_rotation
from .helper_ply import read_ply

from .s3dis_legacy import S3DIS_Pillar_Test, S3DIS_Pillar_TrainVal


class S3DIS_Pillar():

    def compute_mask(self, pt, bs):
        # build the mask
        mask_x = np.logical_and(self.data[:,0]<=pt[0]+bs/2, self.data[:,0]>=pt[0]-bs/2)
        mask_y = np.logical_and(self.data[:,1]<=pt[1]+bs/2, self.data[:,1]>=pt[1]-bs/2)
        mask = np.logical_and(mask_x, mask_y)
        return mask

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

        self.filelist = []
        for area_idx in range(1 ,7):
            folder = os.path.join(self.dataset_dir, f"Area_{area_idx}")
            if (self.split in ['validation', 'test']) and config['dataset']['val_area']==area_idx:
                self.filelist = [os.path.join(f"Area_{area_idx}", dataset) for dataset in os.listdir(folder)]
            elif self.split == 'training' and config['dataset']['val_area']!=area_idx:
                self.filelist = self.filelist + [os.path.join(f"Area_{area_idx}", dataset) for dataset in os.listdir(folder)]
        self.filelist.sort()


        if network_function is not None:
            self.net = network_function()
        else:
            self.net = None


    def size(self):
        return len(self.filelist)

    def get_points(self):
        return self.data[:,:3]

    def get_labels(self):
        return self.data[:, 6].astype(int)


    def __len__(self):
        if self.split == "training":
            return self.cfg['training']['training_steps'] * self.cfg['training']['batch_size']
        elif self.split == "validation":
            return self.cfg['training']['validation_steps'] * self.cfg['training']['batch_size']
        else: # test, requires to have computed the sliding window
            return len(self.choices)


    def compute_sliding_window(self, index, step, npoints):

        # loading the data
        filename_data = os.path.join(self.dataset_dir, self.filelist[index], 'xyzrgb.npy')
        filename_labels = os.path.join(self.dataset_dir, self.filelist[index], 'label.npy')
        data = np.load(filename_data).astype(np.float32)
        labels = np.load(filename_labels).astype(np.float32).flatten()
        labels = np.expand_dims(labels, axis=1)
        self.data = np.concatenate([data, labels], axis=1)

        # compute occupation grid
        mini = self.data[:,:2].min(0)
        discretized = ((self.data[:,:2]-mini).astype(float)/step).astype(int)
        self.pts = np.unique(discretized, axis=0)
        self.pts = self.pts.astype(np.float)*step + mini + step/2

        # compute the masks
        self.choices = []
        self.pts_ref = []
        for index in tqdm(range(self.pts.shape[0]), ncols=80, desc="Pillar computation"):
            pt_ref = self.pts[index]
            mask = self.compute_mask(pt_ref, self.cfg['dataset']['pillar_size'])

            pillar_points_indices = np.where(mask)[0]
            valid_points_indices = pillar_points_indices.copy()

            while(valid_points_indices is not None):
                # print(valid_points_indices.shape[0])
                if valid_points_indices.shape[0] > npoints:
                    choice = np.random.choice(valid_points_indices.shape[0], npoints, replace=True)
                    mask[valid_points_indices[choice]] = False
                    choice = valid_points_indices[choice]
                    valid_points_indices = np.where(mask)[0]
                else:
                    choice = np.random.choice(pillar_points_indices.shape[0], npoints-valid_points_indices.shape[0], replace=True)
                    choice = np.concatenate([valid_points_indices, pillar_points_indices[choice]], axis=0)
                    valid_points_indices = None

                self.choices.append(choice)
                self.pts_ref.append(pt_ref)




    @with_indices_computation_rotation
    def __getitem__(self, index):

        if self.split in ["training", "validation"]:

            index = random.randint(0, len(self.filelist)-1)
            filename_data = os.path.join(self.dataset_dir, self.filelist[index], 'xyzrgb.npy')
            filename_labels = os.path.join(self.dataset_dir, self.filelist[index], 'label.npy')
            data = np.load(filename_data).astype(np.float32)
            labels = np.load(filename_labels).astype(np.float32).flatten()
            labels = np.expand_dims(labels, axis=1)
            data = np.concatenate([data, labels], axis=1)

            # apply transformations on data
            if self.t_data is not None:
                for t in self.t_data:
                    data = t(data)

            # get the features, labels and points
            fts = data[:,3:6]
            lbs = data[:, 6].astype(int)
            pts = data[:, :3]

            choice = 0 # not used at training or validation

        else: # it is a test
            choice = self.choices[index]
            pts = self.data[choice]

            # get the features, labels and points
            fts = pts[:,3:6]
            lbs = pts[:, 6].astype(int)
            pts = pts[:, :3]

        # apply transformations on points
        if self.t_points is not None:
            for t in self.t_points:
                pts = t(pts)

        # apply transformations on features
        fts = fts / 255
        if self.t_features is not None:
            for t in self.t_features:
                fts = t(fts)

        pts = torch.from_numpy(pts).float()
        fts = torch.from_numpy(fts).float()
        lbs = torch.from_numpy(lbs).long()

        pts = pts.transpose(0,1)
        fts = fts.transpose(0,1)

        return_dict = {
            "pts": pts,
            "features": fts,
            "target": lbs,
            "pts_ids": choice
        }

        return return_dict


    @staticmethod
    def get_class_weights(return_torch_tensor=True):
        # pre-calculate the number of points in each category
        raise NotImplementedError


