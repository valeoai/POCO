import torch
from torch._C import Value
import torch.nn as nn

from lightconvpoint.nn.conv_fkaconv_radius import Convolution_FKAConv as Conv
from lightconvpoint.nn import max_pool, interpolate
from lightconvpoint.spatial import radius_nn, knn, sampling_quantized as sampling
from torch_geometric.data import Data

class ResidualBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,  neighborhood_size, max_nn=16, ratio=None,  n_support=None):
        super().__init__()

        self.cv0 = nn.Conv1d(in_channels, in_channels//2, 1)
        self.bn0 = nn.BatchNorm1d(in_channels//2)
        self.cv1 = Conv(in_channels//2, in_channels//2, kernel_size)
        self.bn1 = nn.BatchNorm1d(in_channels//2)
        self.cv2 = nn.Conv1d(in_channels//2, out_channels, 1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU(inplace=True)

        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.bn_shortcut = nn.BatchNorm1d(out_channels) if in_channels != out_channels else nn.Identity()
        
        self.ratio = ratio
        self.n_support = n_support
        assert( (self.ratio is None) != (self.n_support is None))

        self.neighborhood_size = neighborhood_size
        self.max_nn = max_nn
    
    def forward(self, x, pos, support_points, neighbors_indices, mask_indices):

        support_points, _ = sampling(pos, self.ratio, self.n_support, support_points, None)
        neighbors_indices, mask_indices = radius_nn(pos, support_points, self.neighborhood_size, self.max_nn, neighbors_indices, mask_indices)

        if x is not None:

            x_short = x
            x = self.activation(self.bn0(self.cv0(x)))
            x = self.activation(self.bn1(self.cv1(x, pos, support_points, neighbors_indices, mask_indices)))
            x = self.bn2(self.cv2(x))

            x_short = self.bn_shortcut(self.shortcut(x_short))
            if x_short.shape[2] != x.shape[2]:
                x_short = max_pool(x_short, neighbors_indices)

            x = self.activation(x + x_short)

        return x, support_points, neighbors_indices, mask_indices


class FKAConvNetwork(torch.nn.Module):

    def __init__(self, in_channels, out_channels, segmentation=False, hidden=64, dropout=0.5, max_nn=16):
        super().__init__()

        self.lcp_preprocess = True
        self.segmentation = segmentation

        self.cv0 = Conv(in_channels, hidden, 16)
        self.bn0 = nn.BatchNorm1d(hidden)

        self.radius = 2.5 * 0.06
        self.kernel_size = 16

        if isinstance(max_nn, int):
            max_nn = [max_nn for _ in range(9)]
        self.max_nn = max_nn
        
        self.resnetb01 = ResidualBlock(   hidden,    hidden, self.kernel_size,    self.radius, max_nn=self.max_nn[0], ratio=1)
        self.resnetb10 = ResidualBlock(   hidden,  2*hidden, self.kernel_size,    self.radius, max_nn=self.max_nn[1], ratio=0.25)
        self.resnetb11 = ResidualBlock( 2*hidden,  2*hidden, self.kernel_size,  2*self.radius, max_nn=self.max_nn[2], ratio=1) 
        self.resnetb20 = ResidualBlock( 2*hidden,  4*hidden, self.kernel_size,  2*self.radius, max_nn=self.max_nn[3], ratio=0.25)
        self.resnetb21 = ResidualBlock( 4*hidden,  4*hidden, self.kernel_size,  4*self.radius, max_nn=self.max_nn[4], ratio=1)
        self.resnetb30 = ResidualBlock( 4*hidden,  8*hidden, self.kernel_size,  4*self.radius, max_nn=self.max_nn[5], ratio=0.25)
        self.resnetb31 = ResidualBlock( 8*hidden,  8*hidden, self.kernel_size,  8*self.radius, max_nn=self.max_nn[6], ratio=1)
        self.resnetb40 = ResidualBlock( 8*hidden, 16*hidden, self.kernel_size,  8*self.radius, max_nn=self.max_nn[7], ratio=0.25)
        self.resnetb41 = ResidualBlock(16*hidden, 16*hidden, self.kernel_size, 16*self.radius, max_nn=self.max_nn[8], ratio=1)

        if self.segmentation:

            self.cv5 = nn.Conv1d(32*hidden, 16 * hidden, 1)
            self.bn5 = nn.BatchNorm1d(16*hidden)
            self.cv3d = nn.Conv1d(24*hidden, 8 * hidden, 1)
            self.bn3d = nn.BatchNorm1d(8 * hidden)
            self.cv2d = nn.Conv1d(12 * hidden, 4 * hidden, 1)
            self.bn2d = nn.BatchNorm1d(4 * hidden)
            self.cv1d = nn.Conv1d(6 * hidden, 2 * hidden, 1)
            self.bn1d = nn.BatchNorm1d(2 * hidden)
            self.cv0d = nn.Conv1d(3 * hidden, hidden, 1)
            self.bn0d = nn.BatchNorm1d(hidden)
            self.fcout = nn.Conv1d(hidden, out_channels, 1)
        else:

            self.fcout = nn.Linear(1024, out_channels)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, data, spatial_only=False, return_all_decoder_features=False):

        pos = data["pos"]
        x = None if (spatial_only or ("x" not in data) or (data["x"] is None)) else data["x"]

        if len(pos.shape) == 2:
            pos = pos.unsqueeze(0)
        pos = pos.transpose(1,2)

        if x is not None:
            if len(x.shape) == 2:
                x = x.unsqeeze(0)
            x = x.transpose(1,2)

        if self.segmentation:
            if ("net_indices" in data) and (data["net_indices"] is not None):
                ids0, ids10, ids11, ids20, ids21, ids30, ids31, ids40, ids41, ids3u, ids2u, ids1u, ids0u = data["net_indices"]
            else:
                ids0, ids10, ids11, ids20, ids21, ids30, ids31, ids40, ids41, ids3u, ids2u, ids1u, ids0u = [None for _ in range(13)]
        else:
            if ("net_indices" in data) and (data["net_indices"] is not None):
                ids0, ids10, ids11, ids20, ids21, ids30, ids31, ids40, ids41 = data["net_indices"]
            else:
                ids0, ids10, ids11, ids20, ids21, ids30, ids31, ids40, ids41 = [None for _ in range(9)]

        if ("net_mask" in data) and (data["net_mask"] is not None):
            mask0, mask10, mask11, mask20, mask21, mask30, mask31, mask40, mask41 = data["net_mask"]
        else:
            mask0, mask10, mask11, mask20, mask21, mask30, mask31, mask40, mask41 = [None for _ in range(9)]

        if ("net_support" in data) and (data["net_support"] is not None):
            support1, support2, support3, support4 = data["net_support"]
        else:
            support1, support2, support3, support4 = [None for _ in range(4)]



        ids0, mask0 = radius_nn(pos, pos, self.radius, self.max_nn[0], ids0, mask0)
        
        if x is not None:
            x0 = self.activation(self.bn0(self.cv0(x, pos, pos, ids0, mask0)))
        else:
            x0 = None
        
        x0, _, _, _ = self.resnetb01(x0, pos, pos, ids0, mask0)
        x1, support1, ids10, mask10 = self.resnetb10(x0, pos, support1, ids10, mask10)
        x1, _, ids11, mask11 = self.resnetb11(x1, support1, support1, ids11, mask11)
        x2, support2, ids20, mask20 = self.resnetb20(x1, support1, support2, ids20, mask20)
        x2, _, ids21, mask21 = self.resnetb21(x2, support2, support2, ids21, mask21)
        x3, support3, ids30, mask30 = self.resnetb30(x2, support2, support3, ids30, mask30)
        x3, _, ids31, mask31 = self.resnetb31(x3, support3, support3, ids31, mask31)
        x4, support4, ids40, mask40 = self.resnetb40(x3, support3, support4, ids40, mask40)
        x4, _, ids41, mask41 = self.resnetb41(x4, support4, support4, ids41, mask41)
 
        if self.segmentation:
            ids3u = knn(support4, support3, 1, ids3u)
            ids2u = knn(support3, support2, 1, ids2u)
            ids1u = knn(support2, support1, 1, ids1u)
            ids0u = knn(support1, pos, 1, ids0u)

            xout = x4

            if xout is not None:

                x5 = x4.max(dim=2, keepdim=True)[0].expand_as(x4)
                x4d = self.activation(self.bn5(self.cv5(torch.cat([x4, x5], dim=1))))
                
                x3d = interpolate(x4d, ids3u)
                x3d = self.activation(self.bn3d(self.cv3d(torch.cat([x3d, x3], dim=1))))

                x2d = interpolate(x3d, ids2u)
                x2d = self.activation(self.bn2d(self.cv2d(torch.cat([x2d, x2], dim=1))))
                
                x1d = interpolate(x2d, ids1u)
                x1d = self.activation(self.bn1d(self.cv1d(torch.cat([x1d, x1], dim=1))))
                
                xout = interpolate(x1d, ids0u)
                xout = self.activation(self.bn0d(self.cv0d(torch.cat([xout, x0], dim=1))))
                xout = self.dropout(xout)
                xout = self.fcout(xout)
            
                if return_all_decoder_features:
                    xout = [x4d, x3d, x2d, x1d, xout]

            output_data = Data(outputs=xout,
                        net_support=[support1, support2, support3, support4],
                        net_indices = [ids0, ids10, ids11, ids20, ids21, ids30, ids31, ids40, ids41, ids3u, ids2u, ids1u, ids0u],
                        net_mask = [mask0, mask10, mask11, mask20, mask21, mask30, mask31, mask40, mask41])

            if xout is None:
                for support_id, support in enumerate(output_data["net_support"]):
                    output_data["net_support"][support_id] = support.squeeze(0)
                for ids_id, ids in enumerate(output_data["net_indices"]):
                    output_data["net_indices"][ids_id] = ids.squeeze(0)
                for ids_id, ids in enumerate(output_data["net_mask"]):
                    output_data["net_mask"][ids_id] = ids.squeeze(0)

            return output_data

        else:

            xout = x4
            if xout is not None:
                xout = xout.mean(dim=2)
                xout = self.dropout(xout)
                xout = self.fcout(xout)

            output_data = Data(outputs=xout,
                net_support = [support1, support2, support3, support4],
                net_indices = [ids0, ids10, ids11, ids20, ids21, ids30, ids31, ids40, ids41],
                net_mask = [mask0, mask10, mask11, mask20, mask21, mask30, mask31, mask40, mask41])

            if xout is None:
                for support_id, support in enumerate(output_data["net_support"]):
                    output_data["net_support"][support_id] = support.squeeze(0)
                for ids_id, ids in enumerate(output_data["net_indices"]):
                    output_data["net_indices"][ids_id] = ids.squeeze(0)
                for ids_id, ids in enumerate(output_data["net_mask"]):
                    output_data["net_mask"][ids_id] = ids.squeeze(0)

            return output_data