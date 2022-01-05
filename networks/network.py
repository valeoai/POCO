import torch
from torch_geometric.data import Data
import logging
from .backbone import *
from .decoder import *
from lightconvpoint.spatial import knn, sampling_quantized as sampling
from lightconvpoint.utils.functional import batch_gather
from lightconvpoint.nn import max_pool, interpolate

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Network(torch.nn.Module):

    def __init__(self, in_channels, latent_size, out_channels, backbone, decoder, **kwargs):
        super().__init__()

        self.net = eval(backbone)(in_channels, latent_size, segmentation=True, dropout=0)
        self.projection = eval(decoder["name"])(latent_size, out_channels, decoder["k"])
        self.lcp_preprocess = True

        logging.info(f"Network -- backbone -- {count_parameters(self.net)} parameters")
        logging.info(f"Network -- projection -- {count_parameters(self.projection)} parameters")

    def forward(self, data, spatial_only=False, spectral_only=False):

        if spatial_only:
            net_data = self.net(data, spatial_only=spatial_only)
            if "output_support" in net_data:
                data["output_support"] = net_data["output_support"]
            proj_data = self.projection.forward_spatial(data)
            net_data["proj_indices"] = proj_data["proj_indices"]
            return net_data

        if not spectral_only:
            spatial_data = self.net.forward_spatial(data)
            if "output_support" in spatial_data:
                data["output_support"] = spatial_data["output_support"]
            proj_data = self.projection.forward_spatial(data)
            spatial_data["proj_indices"] = proj_data["proj_indices"]
            for key, value in spatial_data.items():
                data[key] = value

        latents = self.net(data, spectral_only=True)
        data["latents"] = latents
        ret_data = self.projection(data, spectral_only=True)

        return ret_data



    def get_latent(self, data, with_correction=False, spatial_only=False, spectral_only=False):

        latents = self.net(data, spatial_only=spatial_only, spectral_only=spectral_only)
        data["latents"] = latents

        data["proj_correction"] = None
        if with_correction:
            data_in_proj = {"latents":latents, "pos":data["pos"], "pos_non_manifold":data["pos"].clone(), "proj_correction":None}
            data_proj = self.projection(data_in_proj, spectral_only=False)
            data["proj_correction"] = data_proj
        return data

    def from_latent(self, data):
        data_proj = self.projection(data)
        return data_proj#["outputs"]


class NetworkMultiScale(torch.nn.Module):

    def __init__(self, in_channels, latent_size, out_channels, backbone, decoder, **kwargs):
        super().__init__()

        self.net = eval(backbone)(in_channels, latent_size, segmentation=True, dropout=0)

        self.merge_latent = torch.nn.Sequential(
            torch.nn.Conv1d(2*latent_size, latent_size,1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(latent_size, latent_size,1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(latent_size, latent_size,1)
        )

        if "Radius" in decoder["name"]:
            self.projection = eval(decoder["name"])(latent_size, out_channels, decoder["radius"])
        else:
            self.projection = eval(decoder["name"])(latent_size, out_channels, decoder["k"])
        self.lcp_preprocess = True

        logging.info(f"Network -- backbone -- {count_parameters(self.net)} parameters")
        logging.info(f"Network -- projection -- {count_parameters(self.projection)} parameters")

    def forward(self, data):

        # compute the down sampled latents
        # ids_down = torch.rand((data["pos"].shape[0], 3000), device=data["pos"].device) * data["pos"].shape[2]
        # ids_down = ids_down.long()

        with torch.no_grad():
            pos_down, idsDown = sampling(data["pos"], n_support=3000)
            x_down = batch_gather(data["x"], dim=2, index=idsDown).contiguous()
            data_down = {'x':x_down, 'pos':pos_down}
            latents_down = self.net(data_down)
            idsUp = knn(pos_down, data["pos"], 1)
            latents_down = interpolate(latents_down, idsUp)
        
        latents = self.net(data)
        
        latents = torch.cat([latents, latents_down], dim=1)
        latents = self.merge_latent(latents)

        data["latents"] = latents
        ret_data = self.projection(data)

        return ret_data

    def train(self, mode=True):
        r"""Sets the module in training mode."""      
        self.training = mode
        # set only the merge to train
        for module in self.children():
            module.train(False)
        self.merge_latent.train(mode)
        return self

    def get_latent(self, data, with_correction=False, spatial_only=False, spectral_only=False):

        latents = self.net(data, spatial_only=spatial_only, spectral_only=spectral_only)
        data["latents"] = latents

        data["proj_correction"] = None
        if with_correction:
            data_in_proj = {"latents":latents, "pos":data["pos"], "pos_non_manifold":data["pos"].clone(), "proj_correction":None}
            data_proj = self.projection(data_in_proj, spectral_only=False)
            data["proj_correction"] = data_proj
        return data

    def from_latent(self, data):
        data_proj = self.projection(data)
        return data_proj#["outputs"]
