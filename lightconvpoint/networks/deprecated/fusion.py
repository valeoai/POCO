import torch
import torch.nn as nn
import torch.nn.functional as F
import lightconvpoint.nn as lcp_nn
import lightconvpoint.networks as lcp_net
import os
from contextlib import nullcontext

class Fusion(nn.Module): # Segsmall with config option for precomputing in the dataloader


    def train(self, mode=True):
        """
        Override the default train() to freeze the backbones
        """
        super(Fusion, self).train(mode)
        if self.freeze: # freeze the encoder
            self.base_network_noc.eval()
            self.base_network_rgb.eval()


    def __init__(self, in_channels, out_channels, ConvNet, Search, **kwargs):
        super().__init__()
        self.ConvNet = ConvNet
        self.Search = Search
        self.in_channels = in_channels
        self.out_channels = out_channels

        if 'config' in kwargs:
            self.config = kwargs['config']
        else:
            raise Exception("Error - config dictionnary needed for fusion")

        # option used only at test time to prevent loading the weights twice
        if 'loadSubModelWeights' in kwargs:
            loadSubModelWeights = kwargs['loadSubModelWeights']
        else:
            loadSubModelWeights = True

        self.base_network_rgb = getattr(lcp_net, self.config["network"]["fusion_submodel"][0])(
                in_channels, out_channels, self.ConvNet, self.Search, **kwargs
            )
        self.base_network_noc = getattr(lcp_net, self.config["network"]["fusion_submodel"][1])(
                in_channels, out_channels, self.ConvNet, self.Search, **kwargs
            )

        if self.config["network"]["fusion_submodeldir"] is not None and loadSubModelWeights:
            self.base_network_rgb.load_state_dict(
                torch.load(
                    os.path.join(
                        self.config["network"]["fusion_submodeldir"][0], "checkpoint.pth"))["state_dict"])
            self.base_network_noc.load_state_dict(
                torch.load(
                    os.path.join(
                        self.config["network"]["fusion_submodeldir"][1], "checkpoint.pth"))["state_dict"])

        self.cv1 = lcp_nn.Conv(
                    self.ConvNet(self.base_network_rgb.features_out_size + self.base_network_noc.features_out_size, 96, 16),
                    self.Search(K=16)
                )
        self.bn1 = nn.BatchNorm1d(96)
        self.cv2 = lcp_nn.Conv(
                    self.ConvNet(96, 48, 16),
                    self.Search(K=16)
                )
        self.bn2 = nn.BatchNorm1d(48)
        self.fc = nn.Conv1d(48 + 2*out_channels, out_channels, 1)
        self.drop = nn.Dropout(0.5)

        self.freeze=True
        if self.freeze:
            self.base_network_noc.eval()
            self.base_network_rgb.eval()

    def forward(
        self, x, input_pts, support_points=None, indices=None):
        
        if x is None:
            _, ids_base, pts_base = self.base_network_rgb(x, input_pts, support_points, indices)
            _, _, idsR = self.cv1(None, input_pts, input_pts)
            return None, ids_base + [idsR], pts_base
        
        with torch.no_grad() if self.freeze else nullcontext():
            outputs_rgb, features_rgb = self.base_network_rgb(x, input_pts, support_points, indices, return_features=True)
            outputs_noc, features_noc = self.base_network_noc(torch.ones_like(x), input_pts, support_points, indices, return_features=True)

        # compute fusion features
        x0 = torch.cat([features_rgb, features_noc], dim=1)
        x1, _, ids1 = self.cv1(x0, input_pts, input_pts, indices=indices[-1])
        x1 = F.relu(self.bn1(x1))
        x2, _, _ = self.cv2(x1, input_pts, input_pts, ids1)
        x2 = F.relu(self.bn2(x2))

        # decision layer
        outputs_fus = torch.cat([outputs_rgb, outputs_noc, x2], dim=1)
        outputs_fus = self.fc(outputs_fus)

        return outputs_fus