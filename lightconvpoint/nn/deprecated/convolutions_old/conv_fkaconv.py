import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from lightconvpoint.nn.deprecated import Module
from lightconvpoint.spatial.deprecated import knn, sampling_quantized
from lightconvpoint.utils.functional import batch_gather
from .convolution import ConvBase

class FKAConv(ConvBase):
    """FKAConv convolution layer.

    To be used with a `lightconvpoint.nn.Conv` instance.

    # Arguments
        in_channels: int.
            The number of input channels.
        out_channels: int.
            The number of output channels.
        kernel_size: int.
            The size of the kernel.
        bias: Boolean.
            Defaults to `False`. Add an optimizable bias.
        dim: int.
            Defaults to `3`. Spatial dimension.

    # Forward arguments
        input: 3-D torch tensor.
            The input features. Dimensions are (B, I, N) with B the batch size, I the
            number of input channels and N the number of input points.
        points: 3-D torch tensor.
            The input points. Dimensions are (B, D, N) with B the batch size, D the
            dimension of the spatial space and N the number of input points.
        support_points: 3-D torch tensor.
            The support points to project features on. Dimensions are (B, O, N) with B
            the batch size, O the number of output channels and N the number of input
            points.

    # Returns
        features: 3-D torch tensor.
            The computed features. Dimensions are (B, O, N) with B the batch size,
            O the number of output channels and N the number of input points.
        support_points: 3-D torch tensor.
            The support points. If they were provided as an input, return the same
            tensor.
    """

    def __init__(self, in_channels, out_channels, kernel_size=16, bias=False, dim=3, kernel_separation=False, 
        sampling=sampling_quantized, neighborhood_search=knn, ratio=1, neighborhood_size=16,
        **kwargs):
        super().__init__(sampling=sampling, neighborhood_search=neighborhood_search, ratio=ratio, neighborhood_size=neighborhood_size, **kwargs)

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

        # normalization radius
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


    def forward_with_features(self, x: torch.Tensor, pos: torch.Tensor, support_points: list, indices:list):
        """Computes the features associated with the support points."""

        assert(isinstance(support_points, list))
        assert(isinstance(indices, list))

        indices = indices[0]
        support_points = support_points[0]

        points = batch_gather(pos, dim=2, index=indices).contiguous()
        input = batch_gather(x, dim=2, index=indices).contiguous()

        # center the neighborhoods (local coordinates)
        pts = points - support_points.unsqueeze(3)

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
        mat = F.relu(self.bn1(self.fc1(pts)))
        mp1 = torch.max(mat * distance_weight, dim=3, keepdim=True)[0].expand(
            (-1, -1, -1, mat.shape[3])
        )
        mat = torch.cat([mat, mp1], dim=1)
        mat = F.relu(self.bn2(self.fc2(mat)))
        mp2 = torch.max(mat * distance_weight, dim=3, keepdim=True)[0].expand(
            (-1, -1, -1, mat.shape[3])
        )
        mat = torch.cat([mat, mp2], dim=1)
        mat = F.relu(self.fc3(mat)) * distance_weight

        # compute features
        features = torch.matmul(
            input.transpose(1, 2), mat.permute(0, 2, 3, 1)
        ).transpose(1, 2)
        features = self.cv(features).squeeze(3)

        return features

