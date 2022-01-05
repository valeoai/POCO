import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil

from torch.nn.modules import distance
# from lightconvpoint.spatial import knn, sampling_quantized
from lightconvpoint.utils.functional import batch_gather
import torch

class Convolution_FKAConv(torch.nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size=16, bias=False, dim=3, kernel_separation=False, **kwargs):
        super().__init__()

        # parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.dim = dim

        # convolution kernel
        if kernel_separation:
            # equivalent to two kernels K1 * K2
            dm = int(ceil(self.out_channels / self.in_channels))
            self.cv = nn.Sequential(
                nn.Conv2d(in_channels, dm*in_channels, (1, kernel_size), bias=bias, groups=self.in_channels),
                nn.Conv2d(in_channels*dm, out_channels, (1, 1), bias=bias)
            )
        else:
            self.cv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), bias=bias)

        # features to kernel weights
        self.fc1 = nn.Conv2d(self.dim, self.kernel_size, 1, bias=False)
        self.fc2 = nn.Conv2d(2 * self.kernel_size, self.kernel_size, 1, bias=False)
        self.fc3 = nn.Conv2d(2 * self.kernel_size, self.kernel_size, 1, bias=False)

    def forward(self, x, pos, support_points, neighbors_indices, radius):
        
        


        # get the mask
        mask = (neighbors_indices > -1)
        neighbors_indices[~mask] = 0

        pos = batch_gather(pos, dim=2, index=neighbors_indices).contiguous()
        x = batch_gather(x, dim=2, index=neighbors_indices).contiguous()

        # center the neighborhoods (local coordinates)
        pts = pos - support_points.unsqueeze(3)

        mask_pts = ~torch.isinf(pts[:,:1])
        pts = torch.nan_to_num(pts, 0.0, 0.0, 0.0)

        # normalize
        pts = pts / radius

        # estimate distance weights
        distance_weight = mask.float()
        distance_weight = F.normalize(distance_weight, dim=2)
        distance_weight = distance_weight.unsqueeze(1)

        distance_weight = distance_weight * mask_pts

        # feature weighting matrix estimation
        mat = F.relu(self.fc1(pts)) * distance_weight
        mp1 = torch.max(mat, dim=3, keepdim=True)[0].expand((-1, -1, -1, mat.shape[3]))
        mat = torch.cat([mat, mp1], dim=1)
        mat = F.relu(self.fc2(mat)) * distance_weight
        mp2 = torch.max(mat, dim=3, keepdim=True)[0].expand((-1, -1, -1, mat.shape[3]))
        mat = torch.cat([mat, mp2], dim=1)
        mat = F.relu(self.fc3(mat)) * distance_weight

        # compute features
        features = torch.matmul(
            x.transpose(1, 2), mat.permute(0, 2, 3, 1)
        ).transpose(1, 2)
        features = self.cv(features).squeeze(3)
        
        
        mask_support = (torch.isinf(support_points[:,:1])).expand_as(features)
        features[mask_support] = float("Inf")
        
        print(features)

        # print(support_points.shape, features.shape)

        return features

