import torch
import torch.nn as nn


class PCCN(nn.Module):
    """PCCN convolution layer.

    Implementation from the paper Deep Parametric Convtinuous Convolutional Neural Network
    (http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Deep_Parametric_Continuous_CVPR_2018_paper.pdf)
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
            The computed features. Dimensions are (B, O, N) with B the batch size, O the
            number of output channels and N the number of input points.
        support_points: 3-D torch tensor.
            The support points. If they were provided as an input, return the same
            tensor.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, dim=3):
        super().__init__()

        # parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.use_bias = bias
        self.dim = dim

        # weight matrix
        self.weight = nn.Parameter(
            torch.Tensor(in_channels, out_channels), requires_grad=True
        )
        torch.nn.init.xavier_uniform_(self.weight.data)

        # bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels, 1), requires_grad=True)
            torch.nn.init.zeros_(self.bias.data)

        # MLP
        modules = []
        proj_dim = self.dim
        modules.append(nn.Linear(proj_dim, 16))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(16, 32))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(32, out_channels))
        self.projector = nn.Sequential(*modules)

    def normalize_points(self, pts, radius=None):
        maxi = torch.sqrt((pts.detach() ** 2).sum(1).max(2)[0])
        maxi = maxi + (maxi == 0)
        return pts / maxi.view(maxi.size(0), 1, maxi.size(1), 1)

    def forward(self, input, points, support_points):
        """Computes the features associated with the support points."""

        # center the neighborhoods (local coordinates)
        pts = points - support_points.unsqueeze(3)

        # normalize points
        pts = self.normalize_points(pts)

        # create the projector
        mat = self.projector(pts.permute(0, 2, 3, 1))

        mat = mat.transpose(2, 3).unsqueeze(4)
        features = torch.matmul(input.permute(0, 2, 3, 1), self.weight)
        features = features.transpose(2, 3).unsqueeze(3)
        features = torch.matmul(features, mat)
        features = features.view(features.shape[:3])
        features = features.transpose(1, 2)

        # add a bias
        if self.use_bias:
            features = features + self.bias

        return features, support_points
