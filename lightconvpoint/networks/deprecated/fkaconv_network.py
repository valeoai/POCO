import torch
import torch.nn as nn
from lightconvpoint.nn.deprecated.module import Module as LCPModule
from lightconvpoint.nn.deprecated.convolutions import FKAConv
from lightconvpoint.nn.deprecated.pooling import max_pool
from lightconvpoint.spatial.deprecated import sampling_quantized, knn, upsample_nearest
from lightconvpoint.utils.functional import batch_gather

class ResidualBlock(LCPModule):

    def __init__(self, in_channels, out_channels, kernel_size, conv_layer, sampling, spatial_search, ratio, neighborhood_size):
        super().__init__()

        self.cv0 = nn.Conv1d(in_channels, in_channels//2, 1)
        self.bn0 = nn.BatchNorm1d(in_channels//2)
        self.cv1 = conv_layer(in_channels//2, in_channels//2, kernel_size, bias=False, sampling=sampling, 
                            spatial_search=spatial_search, ratio=ratio, neighborhood_size=neighborhood_size)
        self.bn1 = nn.BatchNorm1d(in_channels//2)
        self.cv2 = nn.Conv1d(in_channels//2, out_channels, 1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        self.ratio = ratio

    def forward_without_features(self, pos, support_points=None, indices=None):
        return self.cv1(None, pos)

    def forward_with_features(self, x, pos, support_points, indices):
        
        x_short = x
        x = self.activation(self.bn0(self.cv0(x)))
        x = self.activation(self.bn1(self.cv1(x, pos, support_points, indices)))
        x = self.bn2(self.cv2(x))

        if x_short.shape[2] != x.shape[2]:
            x_short = max_pool(x_short, indices)
        x_short = self.shortcut(x_short)

        return self.activation(x + x_short)


class FKAConvNetwork(LCPModule):

    def __init__(self, in_channels, out_channels, segmentation=False, hidden=64, conv_layer=FKAConv ,sampling=sampling_quantized, neighborhood_search=knn):
        super().__init__()

        self.lcp_preprocess = True
        self.segmentation = segmentation

        self.cv0 = conv_layer(in_channels, hidden, 16, sampling=sampling, 
                            neighborhood_search=neighborhood_search, ratio=1, neighborhood_size=16)
        self.bn0 = nn.BatchNorm1d(hidden)


        self.resnetb01 = ResidualBlock(hidden, hidden, 16, conv_layer, sampling, neighborhood_search, 1, 16)

        self.resnetb10 = ResidualBlock(hidden, 2*hidden, 16, conv_layer, sampling, neighborhood_search, 0.25, 16)
        self.resnetb11 = ResidualBlock(2*hidden, 2*hidden, 16, conv_layer, sampling, neighborhood_search, 1, 16) 

        self.resnetb20 = ResidualBlock(2*hidden, 4*hidden, 16, conv_layer, sampling, neighborhood_search, 0.25, 16)
        self.resnetb21 = ResidualBlock(4*hidden, 4*hidden, 16, conv_layer, sampling, neighborhood_search, 1, 16)

        self.resnetb30 = ResidualBlock(4*hidden, 8*hidden, 16, conv_layer, sampling, neighborhood_search, 0.25, 16)
        self.resnetb31 = ResidualBlock(8*hidden, 8*hidden, 16, conv_layer, sampling, neighborhood_search, 1, 16)

        self.resnetb40 = ResidualBlock(8*hidden, 16*hidden, 16, conv_layer, sampling, neighborhood_search, 0.25, 16)
        self.resnetb41 = ResidualBlock(16*hidden, 16*hidden, 16, conv_layer, sampling, neighborhood_search, 1, 16)
        
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

        self.dropout = nn.Dropout(0.5)
        self.activation = nn.ReLU()

    def forward_without_features(self, pos, support_points=None, indices=None):
        
        _, _, ids_conv0 = self.cv0(None, pos)

        _, support1, ids10 = self.resnetb10(None, pos)
        _, _, ids11 = self.resnetb11(None, support1[0])
        _, support2, ids20 = self.resnetb20(None, support1[0])
        _, _, ids21 = self.resnetb21(None, support2[0])
        _, support3, ids30 = self.resnetb30(None, support2[0])
        _, _, ids31 = self.resnetb31(None, support3[0])
        _, support4, ids40 = self.resnetb40(None, support3[0])
        _, _, ids41 = self.resnetb41(None, support4[0])

        support_points = support1 + support2 + support3 + support4
        indices = ids_conv0 + ids10 + ids11 + ids20 + ids21 + ids30 + ids31 + ids40 + ids41

        if self.segmentation:
            ids3u = upsample_nearest(support4[0], support3[0])
            ids2u = upsample_nearest(support3[0], support2[0])
            ids1u = upsample_nearest(support2[0], support1[0])
            ids0u = upsample_nearest(support1[0], pos)
            indices += [ids3u, ids2u, ids1u, ids0u]

        return None, support_points, indices


    def forward_with_features(self, x, pos, support_points=None, indices=None):

        if (support_points is None) or (indices is None):
            _, indices, support_points = self.compute_indices(pos)

        if self.segmentation:
            ids0, ids10, ids11, ids20, ids21, ids30, ids31, ids40, ids41, ids3u, ids2u, ids1u, ids0u = indices
        else:
            ids0, ids10, ids11, ids20, ids21, ids30, ids31, ids40, ids41 = indices
        support1, support2, support3, support4 = support_points

        x0 = self.activation(self.bn0(self.cv0(x, pos, pos, ids0)))
        x0 = self.resnetb01(x0, pos, pos, ids0)
        x1 = self.resnetb10(x0, pos, support1, ids10)
        x1 = self.resnetb11(x1, support1, support1, ids11)
        x2 = self.resnetb20(x1, support1, support2, ids20)
        x2 = self.resnetb21(x2, support2, support2, ids21)
        x3 = self.resnetb30(x2, support2, support3, ids30)
        x3 = self.resnetb31(x3, support3, support3, ids31)
        x4 = self.resnetb40(x3, support3, support4, ids40)
        x4 = self.resnetb41(x4, support4, support4, ids41)

        if self.segmentation:
            x5 = x4.max(dim=2, keepdim=True)[0].expand_as(x4)
            x4 = self.activation(self.bn5(self.cv5(torch.cat([x4, x5], dim=1))))
            xout = batch_gather(x4, 2, ids3u)
            xout = self.activation(self.bn3d(self.cv3d(torch.cat([xout, x3], dim=1))))
            xout = batch_gather(xout, 2, ids2u)
            xout = self.activation(self.bn2d(self.cv2d(torch.cat([xout, x2], dim=1))))
            xout = batch_gather(xout, 2, ids1u)
            xout = self.activation(self.bn1d(self.cv1d(torch.cat([xout, x1], dim=1))))
            xout = batch_gather(xout, 2, ids0u)
            xout = self.activation(self.bn0d(self.cv0d(torch.cat([xout, x0], dim=1))))
            xout = self.dropout(xout)
            xout = self.fcout(xout)
        else:
            xout = x4.mean(dim=2)
            xout = self.dropout(xout)
            xout = self.fcout(xout)

        return xout