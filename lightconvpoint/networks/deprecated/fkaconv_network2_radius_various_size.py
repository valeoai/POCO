import torch
import torch.nn as nn
import torch_geometric
from lightconvpoint import spatial
import lightconvpoint

from lightconvpoint.nn import Convolution_FKAConv_3 as Conv
from lightconvpoint.nn import max_pool, interpolate
from lightconvpoint.spatial import knn, sampling_quantized as sampling
from lightconvpoint.spatial.neighborhood_search.radius import radius_3 as radius_nn
# from lightconvpoint.spatial import radius_nn
from torch_geometric.data import Data
from torch_geometric.transforms import GridSampling

import torch
import torch.nn as nn
from torch.nn import init


class MaskedBatchNorm1d(nn.Module):
    """ A masked version of nn.BatchNorm1d. Only tested for 3D inputs.
        Args:
            num_features: :math:`C` from an expected input of size
                :math:`(N, C, L)`
            eps: a value added to the denominator for numerical stability.
                Default: 1e-5
            momentum: the value used for the running_mean and running_var
                computation. Can be set to ``None`` for cumulative moving average
                (i.e. simple average). Default: 0.1
            affine: a boolean value that when set to ``True``, this module has
                learnable affine parameters. Default: ``True``
            track_running_stats: a boolean value that when set to ``True``, this
                module tracks the running mean and variance, and when set to ``False``,
                this module does not track such statistics and always uses batch
                statistics in both training and eval modes. Default: ``True``
        Shape:
            - Input: :math:`(N, C, L)`
            - input_mask: (N, 1, L) tensor of ones and zeros, where the zeros indicate locations not to use.
            - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(MaskedBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.Tensor(num_features, 1))
            self.bias = nn.Parameter(torch.Tensor(num_features, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.track_running_stats = track_running_stats
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(1,num_features, 1))
            self.register_buffer('running_var', torch.ones(1,num_features, 1))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):
        
        input_mask= (~torch.isinf(input[:,:1]))

        # Calculate the masked mean and variance
        B, C, L = input.shape
        if input_mask is not None and input_mask.shape != (B, 1, L):
            raise ValueError('Mask should have shape (B, 1, L).')
        if C != self.num_features:
            raise ValueError('Expected %d channels but input has %d channels' % (self.num_features, C))
        if input_mask is not None:
            masked = input * input_mask
            n = input_mask.sum()
        else:
            masked = input
            n = B * L
        # Sum
        masked_sum = masked.sum(dim=0, keepdim=True).sum(dim=2, keepdim=True)
        # Divide by sum of mask
        current_mean = masked_sum / n
        current_var = ((masked - current_mean) ** 2).sum(dim=0, keepdim=True).sum(dim=2, keepdim=True) / n
        # Update running stats
        if self.track_running_stats and self.training:
            if self.num_batches_tracked == 0:
                self.running_mean = current_mean
                self.running_var = current_var
            else:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * current_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * current_var
            self.num_batches_tracked += 1
        # Norm the input



        if self.track_running_stats and not self.training:
            normed = (masked - self.running_mean) / (torch.sqrt(self.running_var + self.eps))
        else:
            normed = (masked - current_mean) / (torch.sqrt(current_var + self.eps))
        # Apply affine parameters
        if self.affine:
            normed = normed * self.weight + self.bias
        return normed



class ResidualBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.cv0 = nn.Conv1d(in_channels, in_channels//2, 1)
        self.cv1 = Conv(in_channels//2, in_channels//2, kernel_size)
        self.cv2 = nn.Conv1d(in_channels//2, out_channels, 1)
        # self.bn0 = nn.BatchNorm1d(in_channels//2)
        # self.bn1 = nn.BatchNorm1d(in_channels//2)
        # self.bn2 = nn.BatchNorm1d(out_channels)
        self.bn0 = MaskedBatchNorm1d(in_channels//2)
        self.bn1 = MaskedBatchNorm1d(in_channels//2)
        self.bn2 = MaskedBatchNorm1d(out_channels)
        self.activation = nn.ReLU(inplace=True)

        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        # self.bn_shortcut = nn.BatchNorm1d(out_channels) if in_channels != out_channels else nn.Identity()
        self.bn_shortcut = MaskedBatchNorm1d(out_channels) if in_channels != out_channels else nn.Identity()
        

    
    def forward(self, x, pos, support_points, neighbors_indices, radius):

        if x is not None:
            x_short = x
            x = self.activation(self.bn0(self.cv0(x)))
            x = self.activation(self.bn1(self.cv1(x, pos, support_points, neighbors_indices, radius)))
            x = self.bn2(self.cv2(x))

            x_short = self.bn_shortcut(self.shortcut(x_short))
            if x_short.shape[2] != x.shape[2]:
                x_short = max_pool(x_short, neighbors_indices)

            x = self.activation(x + x_short)

        return x


class FKAConvNetwork(torch.nn.Module):

    def __init__(self, in_channels, out_channels, segmentation=False, hidden=64, dropout=0.5, initial_grid_size=0.02):
        super().__init__()

        self.lcp_preprocess = True
        self.segmentation = segmentation

        self.radius = initial_grid_size * 2.5
        self.kernel_size = 16

        self.cv0 = Conv(in_channels, hidden, self.kernel_size)
        self.bn0 = MaskedBatchNorm1d(hidden)
        self.resnetb01 = ResidualBlock(hidden, hidden, self.kernel_size)
        self.resnetb10 = ResidualBlock(hidden, 2*hidden, self.kernel_size)
        self.resnetb11 = ResidualBlock(2*hidden, 2*hidden, self.kernel_size) 
        self.resnetb20 = ResidualBlock(2*hidden, 4*hidden, self.kernel_size)
        self.resnetb21 = ResidualBlock(4*hidden, 4*hidden, self.kernel_size)
        self.resnetb30 = ResidualBlock(4*hidden, 8*hidden, self.kernel_size)
        self.resnetb31 = ResidualBlock(8*hidden, 8*hidden, self.kernel_size)
        self.resnetb40 = ResidualBlock(8*hidden, 16*hidden, self.kernel_size)
        self.resnetb41 = ResidualBlock(16*hidden, 16*hidden, self.kernel_size)
        
        if self.segmentation:
            self.cv5 = nn.Conv1d(32*hidden, 16 * hidden, 1)
            self.bn5 = MaskedBatchNorm1d(16*hidden)
            self.cv3d = nn.Conv1d(24*hidden, 8 * hidden, 1)
            self.bn3d = MaskedBatchNorm1d(8 * hidden)
            self.cv2d = nn.Conv1d(12 * hidden, 4 * hidden, 1)
            self.bn2d = MaskedBatchNorm1d(4 * hidden)
            self.cv1d = nn.Conv1d(6 * hidden, 2 * hidden, 1)
            self.bn1d = MaskedBatchNorm1d(2 * hidden)
            self.cv0d = nn.Conv1d(3 * hidden, hidden, 1)
            self.bn0d = MaskedBatchNorm1d(hidden)
            self.fcout = nn.Conv1d(hidden, out_channels, 1)
        else:
            self.fcout = nn.Linear(1024, out_channels)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, data, spatial_only=False, spectral_only=False, return_all_decoder_features=False):

        pos = data["pos"]

        squeeze_after_computation=False
        if len(pos.shape) == 2:
            pos = pos.unsqueeze(0)
            squeeze_after_computation = True
        pos = pos.transpose(1,2)


        if not spectral_only:
            # compute the support points
            # support1, ids_supp1 = sampling(pos, 0.25)
            # support2, ids_supp2 = sampling(support1, 0.25)
            # support3, ids_supp3 = sampling(support2, 0.25)
            # support4, ids_supp4 = sampling(support3, 0.25)

            def grid_pad(pos, voxel_size, ratio):
                p = pos.squeeze(0).transpose(0,1).clone()

                npoints = max(1, int(p.shape[0] * ratio))

                m_p = (~torch.isinf(p[:,0])).sum()
                p = p[:m_p]
                p = GridSampling(voxel_size)(Data(pos=p))["pos"]

                if p.shape[0] < npoints:
                    p = torch.cat([p, torch.full((npoints-p.shape[0], p.shape[1]), float("Inf"), device=pos.device)])
                else:
                    p = p[torch.randperm(p.shape[0])[:npoints]]

                p = p.unsqueeze(0).transpose(1,2)
                return p

            support1 = grid_pad(pos, 0.08, 0.25)
            support2 = grid_pad(support1, 0.16, 0.25)
            support3 = grid_pad(support2, 0.32, 0.25)
            support4 = grid_pad(support3, 0.64, 0.25)

            # compute the ids
            ids00 = radius_nn(pos, pos, self.radius, 16)
            ids01 = radius_nn(pos, support1, self.radius, 16)
            ids11 = radius_nn(support1, support1, 2*self.radius, 16)
            ids12 = radius_nn(support1, support2, 2*self.radius, 16)
            ids22 = radius_nn(support2, support2, 4*self.radius, 16)
            ids23 = radius_nn(support2, support3, 4*self.radius, 16)
            ids33 = radius_nn(support3, support3, 8*self.radius, 16)
            ids34 = radius_nn(support3, support4, 8*self.radius, 16)
            ids44 = radius_nn(support4, support4, 16*self.radius, 16)


            if self.segmentation:
                ids43 = knn(support4, support3, 1)
                ids32 = knn(support3, support2, 1)
                ids21 = knn(support2, support1, 1)
                ids10 = knn(support1, pos, 1)

            if squeeze_after_computation:
                support1 = support1.squeeze(0)
                support2 = support2.squeeze(0)
                support3 = support3.squeeze(0)
                support4 = support4.squeeze(0)

                # ids_supp1 = ids_supp1.squeeze(0)
                # ids_supp2 = ids_supp2.squeeze(0)
                # ids_supp3 = ids_supp3.squeeze(0)
                # ids_supp4 = ids_supp4.squeeze(0)

                ids00 = ids00.squeeze(0)
                ids01 = ids01.squeeze(0)
                ids11 = ids11.squeeze(0)
                ids12 = ids12.squeeze(0)
                ids22 = ids22.squeeze(0)
                ids23 = ids23.squeeze(0)
                ids33 = ids33.squeeze(0)
                ids34 = ids34.squeeze(0)
                ids44 = ids44.squeeze(0)

                # ids01 = ids00[ids_supp1]
                # ids12 = ids11[ids_supp2]
                # ids23 = ids22[ids_supp3]
                # ids34 = ids33[ids_supp4]
            
            data["support1"] = support1
            data["support2"] = support2
            data["support3"] = support3
            data["support4"] = support4

            data["ids00"] = ids00
            data["ids01"] = ids01
            data["ids11"] = ids11
            data["ids12"] = ids12
            data["ids22"] = ids22
            data["ids23"] = ids23
            data["ids33"] = ids33
            data["ids34"] = ids34
            data["ids44"] = ids44

            if self.segmentation:
                
                if squeeze_after_computation:
                    ids43 = ids43.squeeze(0)
                    ids32 = ids32.squeeze(0)
                    ids21 = ids21.squeeze(0)
                    ids10 = ids10.squeeze(0)
                
                data["ids43"] = ids43
                data["ids32"] = ids32
                data["ids21"] = ids21
                data["ids10"] = ids10


        if not spatial_only:
            x = data["x"].transpose(1,2)
            pos = data["pos"].transpose(1,2)


            # x0 = self.activation(self.bn0(self.cv0(x, pos, pos, data["ids00"], self.radius)))
            x0 = self.cv0(x, pos, pos, data["ids00"], self.radius)

            x0 = self.resnetb01(x0, pos, pos, data["ids00"], self.radius)
            x1 = self.resnetb10(x0, pos, data["support1"], data["ids01"], self.radius)
            x1 = self.resnetb11(x1, data["support1"], data["support1"], data["ids11"], 2*self.radius)
            x2 = self.resnetb20(x1, data["support1"], data["support2"], data["ids12"], 2*self.radius)
            x2 = self.resnetb21(x2, data["support2"], data["support2"], data["ids22"], 4*self.radius)
            x3 = self.resnetb30(x2, data["support2"], data["support3"], data["ids23"], 4*self.radius)
            x3 = self.resnetb31(x3, data["support3"], data["support3"], data["ids33"], 8*self.radius)
            x4 = self.resnetb40(x3, data["support3"], data["support4"], data["ids34"], 8*self.radius)
            x4 = self.resnetb41(x4, data["support4"], data["support4"], data["ids44"], 16*self.radius)

            if self.segmentation:
                
                x5 = x4.max(dim=2, keepdim=True)[0].expand_as(x4)
                x4d = self.activation(self.bn5(self.cv5(torch.cat([x4, x5], dim=1))))
                
                x3d = interpolate(x4d, data["ids43"])
                x3d = self.activation(self.bn3d(self.cv3d(torch.cat([x3d, x3], dim=1))))

                x2d = interpolate(x3d, data["ids32"])
                x2d = self.activation(self.bn2d(self.cv2d(torch.cat([x2d, x2], dim=1))))
                
                x1d = interpolate(x2d, data["ids21"])
                x1d = self.activation(self.bn1d(self.cv1d(torch.cat([x1d, x1], dim=1))))
                
                xout = interpolate(x1d, data["ids10"])
                xout = self.activation(self.bn0d(self.cv0d(torch.cat([xout, x0], dim=1))))
                xout = self.dropout(xout)
                xout = self.fcout(xout)

            else:

                xout = x4.mean(dim=2)
                xout = self.dropout(xout)
                xout = self.fcout(xout)

            return xout

        return data