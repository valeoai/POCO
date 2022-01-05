import torch
import torch.nn as nn
from lightconvpoint.spatial import knn
from lightconvpoint.utils.functional import batch_gather
from torch_geometric.data import Data
import logging
import time

class InterpMaxNet(torch.nn.Module):

    def __init__(self, latent_size, out_channels, K=16):
        super().__init__()

        logging.info(f"InterpNet - Max - K={K}")
        # self.projection_layer = FKAConv(latent_size, latent_size, 16, sampling=None, neighborhood_search=knn, neighborhood_size=16, ratio=1)
        # self.fc1 = torch.nn.Conv2d(latent_size+3, latent_size, 1)
        # self.fc2 = torch.nn.Conv2d(latent_size, latent_size, 1)
        # self.fc3 = torch.nn.Conv2d(latent_size, latent_size, 1)
        # self.fc8 = torch.nn.Conv1d(latent_size, out_channels, 1)
        # self.activation = torch.nn.ReLU()

        self.fc_in = torch.nn.Conv2d(latent_size+3, latent_size, 1)
        mlp_layers = [torch.nn.Conv2d(latent_size, latent_size, 1) for _ in range(2)]
        self.mlp_layers = nn.ModuleList(mlp_layers)
        self.fc_out = torch.nn.Conv1d(latent_size, out_channels, 1)
        self.activation = torch.nn.ReLU()

        self.k = K

    def forward_spatial(self, data):

        pos = data["pos"]
        pos_non_manifold = data["pos_non_manifold"]

        add_batch_dimension_pos = False
        if len(pos.shape) == 2:
            pos = pos.unsqueeze(0)
            add_batch_dimension_pos = True

        add_batch_dimension_non_manifold = False
        if len(pos_non_manifold.shape) == 2:
            pos_non_manifold = pos_non_manifold.unsqueeze(0)
            add_batch_dimension_non_manifold = True

        if pos.shape[1] != 3:
            pos = pos.transpose(1,2)

        if pos_non_manifold.shape[1] != 3:
            pos_non_manifold = pos_non_manifold.transpose(1,2)

        indices = knn(pos, pos_non_manifold, self.k)

        if add_batch_dimension_non_manifold or add_batch_dimension_pos:
            indices = indices.squeeze(0)

        ret_data = {}
        ret_data["proj_indices"] = indices

        return ret_data


    def forward(self, data, spatial_only=False, spectral_only=False):
        if spatial_only:
            return self.forward_spatial(data)

        if not spectral_only:
            spatial_data = self.forward_spatial(data)
            for key, value in spatial_data.items():
                data[key] = value

        
        x = data["latents"]
        indices = data["proj_indices"]
        pos = data["pos"]
        pos_non_manifold = data["pos_non_manifold"]


        if pos.shape[1] != 3:
            pos = pos.transpose(1,2)

        if pos_non_manifold.shape[1] != 3:
            pos_non_manifold = pos_non_manifold.transpose(1,2)

        x = batch_gather(x, 2, indices)
        pos = batch_gather(pos, 2, indices)
        pos = pos_non_manifold.unsqueeze(3) - pos

        x = torch.cat([x,pos], dim=1)
        x = self.fc_in(x)
        for i, l in enumerate(self.mlp_layers):
            x = l(self.activation(x))

        x = x.max(dim=3)[0]
        x = self.fc_out(x)

        return x