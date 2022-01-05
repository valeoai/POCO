import torch
import torch.nn as nn
from lightconvpoint.nn.module import Module as LCPModule
from lightconvpoint.nn.convolutions import PointNet
from lightconvpoint.spatial import sampling_furthest, knn, upsample_nearest
from lightconvpoint.utils.functional import batch_gather



class PointNetPP(LCPModule):

    def __init__(self, in_channels, out_channels, segmentation=False, conv_layer=PointNet, 
                    sampling=sampling_furthest, neighborhood_search=knn):
        super().__init__()

        self.segmentation = segmentation

        self.cv0 = conv_layer(in_channels, mlp=[32,32, 64], sampling=sampling_furthest, neighborhood_search=knn, ratio=1, neighborhood_size=16)
        self.cv1 = conv_layer(64, mlp=[64,64,128], sampling=sampling_furthest, neighborhood_search=knn, ratio=0.25, neighborhood_size=16)
        self.cv2 = conv_layer(128, mlp=[128, 128, 256], sampling=sampling_furthest, neighborhood_search=knn, ratio=0.25, neighborhood_size=16)
        self.cv3 = conv_layer(256, mlp=[256, 256, 512], sampling=sampling_furthest, neighborhood_search=knn, ratio=0.25, neighborhood_size=16)

        
        if self.segmentation:

            # self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 9 + 3, [32, 32, 64], False)
            # self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
            # self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
            # self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
            # self.fp4 = PointNetFeaturePropagation(768, [256, 256])
            # self.fp3 = PointNetFeaturePropagation(384, [256, 256])
            # self.fp2 = PointNetFeaturePropagation(320, [256, 128])
            # self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
            # self.conv1 = nn.Conv1d(128, 128, 1)
            # self.bn1 = nn.BatchNorm1d(128)
            # self.drop1 = nn.Dropout(0.5)
            # self.conv2 = nn.Conv1d(128, num_classes, 1)

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

            self.fc1 = nn.Linear(512, 512)
            self.bn1 = nn.BatchNorm1d(512)
            self.drop1 = nn.Dropout(0.4)
            self.fc2 = nn.Linear(512, 256)
            self.bn2 = nn.BatchNorm1d(256)
            self.drop2 = nn.Dropout(0.4)
            self.fcout = nn.Linear(256, out_channels)

        self.dropout = nn.Dropout(0.5)
        self.activation = nn.ReLU()

    def forward_without_features(self, pos, support_points=None, indices=None):
        
        _, support0, ids0 = self.cv0(None, pos)
        _, support1, ids1 = self.cv1(None, support0[0])
        _, support2, ids2 = self.cv2(None, support1[0])
        _, support3, ids3 = self.cv3(None, support2[0])

        support_points = support0 + support1 + support2 + support3
        indices = ids0 + ids1 + ids2 + ids3

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
            ids0, ids1, ids2, ids3 = indices
        support0, support1, support2, support3 = support_points

        x0 = self.activation(self.cv0(x, pos, support0, ids0))
        x1 = self.activation(self.cv1(x0, support0, support1, ids1))
        x2 = self.activation(self.cv2(x1, support1, support2, ids2))
        x3 = self.activation(self.cv3(x2, support2, support3, ids3))

        if self.segmentation:
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

            xout = x3.max(dim=2)[0]
            xout = self.drop1(self.activation(self.bn1(self.fc1(xout))))
            xout = self.drop2(self.activation(self.bn2(self.fc2(xout))))
            xout = self.fcout(xout)

        return xout