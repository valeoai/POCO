import os
import subprocess
import open3d as o3d
import os
import logging
import torch
from torch_geometric.data import Dataset, Data
import importlib
from pathlib import Path
import numpy as np
import trimesh

point_density = 20
data_dir = "data/SceneNet"
target_dir = f"data/SceneNet{point_density}"

filenames = [
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

for filename in filenames:

    mesh = trimesh.load(os.path.join(data_dir, filename))
    target_fname = os.path.join(target_dir, filename+".xyz")
    
    area = mesh.area
    n_points = int(area * point_density)

    pos, face_index = trimesh.sample.sample_surface(mesh, n_points)
    nls = mesh.face_normals[face_index]

    pos = pos.astype(np.float16)
    nls = pos.astype(np.float16)

    pts = np.concatenate([pos, nls], axis=1)

    # create the directory
    os.makedirs(os.path.dirname(target_fname), exist_ok=True)
    np.savetxt(target_fname, pts)
