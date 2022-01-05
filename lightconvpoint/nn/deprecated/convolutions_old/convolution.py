import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from lightconvpoint.nn.deprecated import Module
from lightconvpoint.spatial.deprecated import knn, sampling_quantized
from lightconvpoint.utils.functional import batch_gather

class ConvBase(Module):
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

    def __init__(self, sampling=sampling_quantized, neighborhood_search=knn, ratio=1, neighborhood_size=16, **kwargs):
        super().__init__()


        # spatial part of the module
        self.sampling = sampling
        self.neighborhood_search = neighborhood_search
        self.neighborhood_size = neighborhood_size
        self.ratio = ratio



    def forward_without_features(self, pos, support_points=None, indices=None):
        if support_points is not None:
            assert(isinstance(support_points, list))
            ids = self.neighborhood_search(pos, support_points[0], self.neighborhood_size)
            return None, support_points, [ids]
        else:
            if self.ratio == 1:
                ids = self.neighborhood_search(pos, pos, self.neighborhood_size)
                return None, [pos], [ids]
            else:
                _, support_points = self.sampling(pos, ratio=self.ratio, return_support_points=True)
                ids = self.neighborhood_search(pos, support_points, self.neighborhood_size)
                return None, [support_points], [ids]

    def forward_with_features(self, x: torch.Tensor, pos: torch.Tensor, support_points: list, indices:list):
        """Computes the features associated with the support points."""

        raise NotImplementedError

