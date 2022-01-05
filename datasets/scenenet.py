import os
import logging
import torch
from torch_geometric.data import Dataset, Data
import importlib
from pathlib import Path
import numpy as np
import trimesh

class SceneNet(Dataset):

    
    def __init__(self,
                 root,
                 train=True,
                 transform=None, split="training", filter_name=None, dataset_size=None, 
                 point_density=None,
                 **kwargs):


        super().__init__(root, transform, None)

        logging.info("Dataset - SceneNet")

        self.split = split
        self.point_density = point_density

        self.filenames = [
        "1Bathroom/107_labels.obj.ply",
        "1Bathroom/1_labels.obj.ply",
        "1Bathroom/28_labels.obj.ply",
        "1Bathroom/29_labels.obj.ply",
        "1Bathroom/4_labels.obj.ply",
        "1Bathroom/5_labels.obj.ply",
        "1Bathroom/69_labels.obj.ply",
        "1Bedroom/3_labels.obj.ply",
        "1Bedroom/77_labels.obj.ply",
        "1Bedroom/bedroom27.obj.ply",
        "1Bedroom/bedroom_1.obj.ply",
        "1Bedroom/bedroom_68.obj.ply",
        "1Bedroom/bedroom_wenfagx.obj.ply",
        "1Bedroom/bedroom_xpg.obj.ply",
        "1Kitchen/1-14_labels.obj.ply",
        "1Kitchen/102.obj.ply",
        "1Kitchen/13_labels.obj.ply",
        "1Kitchen/2.obj.ply",
        "1Kitchen/35_labels.obj.ply",
        "1Kitchen/kitchen_106_blender_name_and_mat.obj.ply",
        "1Kitchen/kitchen_16_blender_name_and_mat.obj.ply",
        "1Kitchen/kitchen_76_blender_name_and_mat.obj.ply",
        "1Living-room/cnh_blender_name_and_mat.obj.ply",
        "1Living-room/living_room_33.obj.ply",
        "1Living-room/lr_kt7_blender_scene.obj.ply",
        "1Living-room/pg_blender_name_and_mat.obj.ply",
        "1Living-room/room_89_blender.obj.ply",
        "1Living-room/room_89_blender_no_paintings.obj.ply",
        "1Living-room/yoa_blender_name_mat.obj.ply",
        "1Office/2_crazy3dfree_labels.obj.ply",
        "1Office/2_hereisfree_labels.obj.ply",
        "1Office/4_3dmodel777.obj.ply",
        "1Office/4_hereisfree_labels.obj.ply",
        "1Office/7_crazy3dfree_old_labels.obj.ply",
        ]
        self.filenames = [os.path.join(self.root, filename) for filename in self.filenames]
        self.filenames.sort()

        self.dataset_size = dataset_size
        if self.dataset_size is not None:
            self.filenames = self.filenames[:self.dataset_size]

        logging.info(f"Dataset - len {len(self.filenames)}")

    def _download(self): # override _download to remove makedirs
        pass

    def download(self):
        pass

    def _process(self):
        pass

    def len(self):
        return len(self.filenames)

    def get_category(self, idx):
        return self.filenames[idx].split("/")[-2]

    def get_object_name(self, idx):
        return self.filenames[idx].split("/")[-1]

    def get_class_name(self, idx):
        return "n/a"

    
    def get_data_for_evaluation(self, idx):
        raise NotImplementedError
        scene = self.filenames[idx]
        input_pointcloud = np.load(scene)
        return input_pointcloud, None


    def get(self, idx):
        """Get item."""

        # load the mesh
        scene_filename = self.filenames[idx]

        data = np.loadtxt(scene_filename+".xyz", dtype=np.float32)

        pos = data[:,:3]
        nls = data[:,3:]

        pos = torch.tensor(pos, dtype=torch.float)
        nls = torch.tensor(nls, dtype=torch.float)
        pos_non_manifold = torch.zeros((1,3), dtype=torch.float)


        data = Data(shape_id=idx, x=torch.ones_like(pos),
                    normal=nls,
                    pos=pos, pos_non_manifold=pos_non_manifold
                    )

        return data