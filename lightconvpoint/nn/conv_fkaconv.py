import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
# from lightconvpoint.spatial import knn, sampling_quantized
from lightconvpoint.utils.functional import batch_gather
import torch

class Convolution_FKAConv(torch.nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size=16, bias=False, dim=3, kernel_separation=False, adaptive_normalization=True,**kwargs):
        super().__init__()

        # parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.dim = dim
        self.adaptive_normalization = adaptive_normalization

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

        # normalization radius
        if self.adaptive_normalization:
            self.norm_radius_momentum = 0.1
            self.norm_radius = nn.Parameter(torch.Tensor(1,), requires_grad=False)
            self.alpha = nn.Parameter(torch.Tensor(1,), requires_grad=True)
            self.beta = nn.Parameter(torch.Tensor(1,), requires_grad=True)
            torch.nn.init.ones_(self.norm_radius.data)
            torch.nn.init.ones_(self.alpha.data)
            torch.nn.init.ones_(self.beta.data)

        # features to kernel weights
        self.fc1 = nn.Conv2d(self.dim, self.kernel_size, 1, bias=False)
        self.fc2 = nn.Conv2d(2 * self.kernel_size, self.kernel_size, 1, bias=False)
        self.fc3 = nn.Conv2d(2 * self.kernel_size, self.kernel_size, 1, bias=False)
        self.bn1 = nn.InstanceNorm2d(self.kernel_size, affine=True)
        self.bn2 = nn.InstanceNorm2d(self.kernel_size, affine=True)



    def fixed_normalization(self, pts, radius=None):
        maxi = torch.sqrt((pts.detach() ** 2).sum(1).max(2)[0])
        maxi = maxi + (maxi == 0)
        return pts / maxi.view(maxi.size(0), 1, maxi.size(1), 1)



    def forward(self, x, pos, support_points, neighbors_indices):

        if x is None:
            return None

        pos = batch_gather(pos, dim=2, index=neighbors_indices).contiguous()
        x = batch_gather(x, dim=2, index=neighbors_indices).contiguous()

        # center the neighborhoods (local coordinates)
        pts = pos - support_points.unsqueeze(3)


        # normalize points
        if self.adaptive_normalization:


            # compute distances from points to their support point
            distances = torch.sqrt((pts.detach() ** 2).sum(1))

            # update the normalization radius
            if self.training:
                mean_radius = distances.max(2)[0].mean()
                self.norm_radius.data = (
                    self.norm_radius.data * (1 - self.norm_radius_momentum)
                    + mean_radius * self.norm_radius_momentum
                )

            # normalize
            pts = pts / self.norm_radius

            # estimate distance weights
            distance_weight = torch.sigmoid(-self.alpha * distances + self.beta)
            distance_weight_s = distance_weight.sum(2, keepdim=True)
            distance_weight_s = distance_weight_s + (distance_weight_s == 0) + 1e-6
            distance_weight = (
                distance_weight / distance_weight_s * distances.shape[2]
            ).unsqueeze(1)

            # feature weighting matrix estimation
            if pts.shape[3] == 1:
                mat = F.relu(self.fc1(pts))
            else:
                mat = F.relu(self.bn1(self.fc1(pts)))
            mp1 = torch.max(mat * distance_weight, dim=3, keepdim=True)[0].expand(
                (-1, -1, -1, mat.shape[3])
            )
            mat = torch.cat([mat, mp1], dim=1)
            if pts.shape[3] == 1:
                mat = F.relu(self.fc2(mat))
            else:
                mat = F.relu(self.bn2(self.fc2(mat)))
            mp2 = torch.max(mat * distance_weight, dim=3, keepdim=True)[0].expand(
                (-1, -1, -1, mat.shape[3])
            )
            mat = torch.cat([mat, mp2], dim=1)
            mat = F.relu(self.fc3(mat)) * distance_weight
            # mat = torch.sigmoid(self.fc3(mat)) * distance_weight
        else:
            pts = self.fixed_normalization(pts)

            # feature weighting matrix estimation
            if pts.shape[3] == 1:
                mat = F.relu(self.fc1(pts))
            else:
                mat = F.relu(self.bn1(self.fc1(pts)))
            mp1 = torch.max(mat, dim=3, keepdim=True)[0].expand(
                (-1, -1, -1, mat.shape[3])
            )
            mat = torch.cat([mat, mp1], dim=1)
            if pts.shape[3] == 1:
                mat = F.relu(self.fc2(mat))
            else:
                mat = F.relu(self.bn2(self.fc2(mat)))
            mp2 = torch.max(mat, dim=3, keepdim=True)[0].expand(
                (-1, -1, -1, mat.shape[3])
            )
            mat = torch.cat([mat, mp2], dim=1)
            mat = F.relu(self.fc3(mat))

        # compute features
        features = torch.matmul(
            x.transpose(1, 2), mat.permute(0, 2, 3, 1)
        ).transpose(1, 2)
        features = self.cv(features).squeeze(3)

        return features

