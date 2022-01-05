import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from lightconvpoint.nn import Module
from lightconvpoint.spatial import knn, sampling_quantized
from lightconvpoint.utils.functional import batch_gather

class PointNet(Module):

    def __init__(self, in_channels,  mlp=None,
        sampling=sampling_quantized, neighborhood_search=knn, ratio=1, neighborhood_size=16,
        **kwargs):
        super().__init__()

        
        # parameters
        self.out_channels = mlp[-1]
        mlp = mlp[:-1]
        self.in_channels = in_channels

        layers = []
        channels = in_channels + 3
        for s in mlp:
            layers.append(nn.Conv2d(channels, s, 1))
            layers.append(nn.ReLU())
            channels = s
        layers.append(nn.Conv2d(s, self.out_channels, 1))

        self.net = nn.Sequential(*layers)

        # spatial part of the module
        self.sampling = sampling
        self.neighborhood_search = neighborhood_search
        self.neighborhood_size = neighborhood_size
        self.ratio = ratio

    def forward_without_features(self, pos, support_points=None, indices=None):
        if self.ratio == 1:
            ids = self.neighborhood_search(pos, pos, self.neighborhood_size)
            return None, [pos], [ids]
        else:
            if support_points is None:
                _, support_points = self.sampling(pos, ratio=self.ratio, return_support_points=True)
            ids = self.neighborhood_search(pos, support_points, self.neighborhood_size)
            return None, [support_points], [ids]

    def forward_with_features(self, x, pos, support_points, indices):
        """Computes the features associated with the support points."""

        input = batch_gather(x, dim=2, index=indices).contiguous()
        
        points = batch_gather(pos, dim=2, index=indices).contiguous()
        points = points - support_points.unsqueeze(3)

        features = self.net(torch.cat([input,points], dim=1))
        features = features.max(dim=3)[0]

        return features

