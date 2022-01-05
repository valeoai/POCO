import torch
import torch.nn as nn
from lightconvpoint.nn import Convolution_ConvPoint as Conv
from lightconvpoint.nn import max_pool, interpolate
from lightconvpoint.spatial import knn, sampling_convpoint as sampling


class ConvPointNetwork(torch.nn.Module):

    def __init__(self, in_channels, out_channels, segmentation=False, hidden=64):
        super().__init__()

        self.segmentation = segmentation
        self.lcp_preprocess = True

        if self.segmentation:

            self.cv0 = Conv(in_channels, hidden, 16, bias=False,)
            self.bn0 = nn.BatchNorm1d(hidden)      
            self.cv1 = Conv(hidden, hidden, 16, bias=False)
            self.bn1 = nn.BatchNorm1d(hidden)      
            self.cv2 = Conv(hidden, hidden, 16, bias=False)
            self.bn2 = nn.BatchNorm1d(hidden)      
            self.cv3 = Conv(hidden, hidden, 16, bias=False)
            self.bn3 = nn.BatchNorm1d(hidden)      
            self.cv4 = Conv(hidden, 2*hidden, 16, bias=False)
            self.bn4 = nn.BatchNorm1d(2*hidden)      
            self.cv5 = Conv(2*hidden, 2*hidden, 16, bias=False)
            self.bn5 = nn.BatchNorm1d(2*hidden)     
            self.cv6 = Conv(2*hidden, 2*hidden, 16, bias=False)
            self.bn6 = nn.BatchNorm1d(2*hidden)

            self.cv5d = Conv(2*hidden, 2*hidden, 16, bias=False)
            self.bn5d = nn.BatchNorm1d(2*hidden)     
            self.cv4d = Conv(4*hidden, 2*hidden, 16, bias=False)
            self.bn4d = nn.BatchNorm1d(2*hidden)     
            self.cv3d = Conv(4*hidden, hidden, 16, bias=False)
            self.bn3d = nn.BatchNorm1d(hidden)      
            self.cv2d = Conv(2*hidden, hidden, 16, bias=False)
            self.bn2d = nn.BatchNorm1d(hidden)
            self.cv1d = Conv(2*hidden, hidden, 16, bias=False)
            self.bn1d = nn.BatchNorm1d(hidden)
            self.cv0d = Conv(2*hidden, hidden, 16, bias=False)
            self.bn0d = nn.BatchNorm1d(hidden)
            self.fcout = nn.Conv1d(2*hidden, out_channels, 1)

        else:

            self.cv1 = Conv(in_channels, hidden, 16, bias=False, sampling=sampling)
            self.bn1 = nn.BatchNorm1d(hidden)            
            self.cv2 = Conv(hidden, 2*hidden, 16, bias=False, sampling=sampling)
            self.bn2 = nn.BatchNorm1d(2*hidden)
            self.cv3 = Conv(2*hidden, 4*hidden, 16, bias=False, sampling=sampling)
            self.bn3 = nn.BatchNorm1d(4*hidden)
            self.cv4 = Conv(4*hidden, 4*hidden, 16, bias=False, sampling=sampling)
            self.bn4 = nn.BatchNorm1d(4*hidden)
            self.cv5 = Conv(4*hidden, 8*hidden, 16, bias=False, sampling=sampling)
            self.bn5 = nn.BatchNorm1d(8*hidden)
            self.fcout = nn.Linear(8*hidden, out_channels)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, pos, support_points=None, neighbors_indices=None):

        if self.segmentation:

            if support_points is not None:
                support1, support2, support3, support4, support5, support6 = support_points
            else:
                support1, support2, support3, support4, support5, support6 = [None for _ in range(6)]

            if neighbors_indices is not None:
                ids0, ids1, ids2, ids3, ids4, ids5, ids6, ids5d, ids4d, ids3d, ids2d, ids1d, ids0d = neighbors_indices
            else:
                ids0, ids1, ids2, ids3, ids4, ids5, ids6, ids5d, ids4d, ids3d, ids2d, ids1d, ids0d = [None for _ in range(13)]

            support1, _ = sampling(pos, 0.25, support1, None)
            ids1 = knn(pos, support1, 16, ids1)
            support2, _ = sampling(support1, 0.25, support2, None)
            ids2 = knn(support1, support2, 16, ids2)
            support3, _ = sampling(support2, 0.25, support3, None)
            ids3 = knn(support2, support3, 16, ids3)
            support4, _ = sampling(support3, 0.25, support4, None)
            ids4 = knn(support3, support4, 16, ids4)
            support5, _ = sampling(support4, 0.25, support5, None)
            ids5 = knn(support4, support5, 16, ids5)
            support6, _ = sampling(support5, 0.25, support6, None)
            ids6 = knn(support5, support6, 16, ids6)
            ids5d = knn(support6, support5, 4, ids5d)
            ids4d = knn(support5, support4, 4, ids4d)
            ids3d = knn(support4, support3, 4, ids3d)
            ids2d = knn(support3, support2, 8, ids2d)
            ids1d = knn(support2, support1, 8, ids1d)
            ids0d = knn(support1, pos, 8, ids0d)
            ids0 = knn(pos, pos, 16, ids0)

            if x is not None:
                x0 = self.activation(self.bn0(self.cv0(x, pos, pos, ids0)))
                x1 = self.activation(self.bn1(self.cv1(x0, pos, support1, ids1)))
                x2 = self.activation(self.bn2(self.cv2(x1, support1, support2, ids2)))
                x3 = self.activation(self.bn3(self.cv3(x2, support2, support3, ids3)))
                x4 = self.activation(self.bn4(self.cv4(x3, support3, support4, ids4)))
                x5 = self.activation(self.bn5(self.cv5(x4, support4, support5, ids5)))
                x6 = self.activation(self.bn6(self.cv6(x5, support5, support6, ids6)))
                x = self.activation(self.bn5d(self.cv5d(x6, support6, support5, ids5d)))
                x = torch.cat([x, x5], dim=1)
                x = self.activation(self.bn4d(self.cv4d(x, support5, support4, ids4d)))
                x = torch.cat([x, x4], dim=1)
                x = self.activation(self.bn3d(self.cv3d(x, support4, support3, ids3d)))
                x = torch.cat([x, x3], dim=1)
                x = self.activation(self.bn2d(self.cv2d(x, support3, support2, ids2d)))
                x = torch.cat([x, x2], dim=1)
                x = self.activation(self.bn1d(self.cv1d(x, support2, support1, ids1d)))
                x = torch.cat([x, x1], dim=1)
                x = self.activation(self.bn0d(self.cv0d(x, support1, pos, ids0d)))
                x = torch.cat([x, x0], dim=1)
                x = self.dropout(x)
                x = self.fcout(x)

            return x, [support1, support2, support3, support4, support5, support6], [ids0, ids1, ids2, ids3, ids4, ids5, ids6, ids5d, ids4d, ids3d, ids2d, ids1d, ids0d]

        else:

            if support_points is not None:
                support1, support2, support3, support4, support5 = support_points
            else:
                support1, support2, support3, support4, support5 = [None for _ in range(5)]

            if neighbors_indices is not None:
                ids1, ids2, ids3, ids4, ids5 = neighbors_indices
            else:
                ids1, ids2, ids3, ids4, ids5 = [None for _ in range(5)]

            support1, _ = sampling(pos, 0.25, support1, None)
            ids1 = knn(pos, support1, 16, ids1)
            support2, _ = sampling(support1, 0.25, support2, None)
            ids2 = knn(support1, support2, 16, ids2)
            support3, _ = sampling(support2, 0.25, support3, None)
            ids3 = knn(support2, support3, 16, ids3)
            support4, _ = sampling(support3, 0.25, support4, None)
            ids4 = knn(support3, support4, 16, ids4)
            support5, _ = sampling(support4, 0.25, support5, None)
            ids5 = knn(support4, support5, 16, ids5)

            if x is not None:

                x = self.activation(self.bn1(self.cv1(x, pos, support1, ids1)))
                x = self.activation(self.bn2(self.cv2(x, support1, support2, ids2)))
                x = self.activation(self.bn3(self.cv3(x, support2, support3, ids3)))
                x = self.activation(self.bn4(self.cv4(x, support3, support4, ids4)))
                x = self.activation(self.bn5(self.cv5(x, support4, support5, ids5)))
                x = x.mean(dim=2)
                x = self.dropout(x)
                x = self.fcout(x)

            return x, [support1, support2, support3, support4, support5], [ids1, ids2, ids3, ids4, ids5]



    # def forward_without_features(self, pos, support_points=None, indices=None):
    #     if self.segmentation:
    #         _, _, ids0 = self.cv0(None, pos)
    #         _, support1, ids1 = self.cv1(None, pos)
    #         _, support2, ids2 = self.cv2(None, support1[0])
    #         _, support3, ids3 = self.cv3(None, support2[0])
    #         _, support4, ids4 = self.cv4(None, support3[0])
    #         _, support5, ids5 = self.cv5(None, support4[0])
    #         _, support6, ids6 = self.cv6(None, support5[0])

    #         _, _, ids5d = self.cv5d(None, support6[0], support5[0])
    #         _, _, ids4d = self.cv4d(None, support5[0], support4[0])
    #         _, _, ids3d = self.cv3d(None, support4[0], support3[0])
    #         _, _, ids2d = self.cv2d(None, support3[0], support2[0])
    #         _, _, ids1d = self.cv1d(None, support2[0], support1[0])
    #         _, _, ids0d = self.cv0d(None, support1[0], pos)

    #         support_points = support1 + support2 + support3 + support4 + support5 + support6
    #         indices = ids0 + ids1 + ids2 + ids3 + ids4 + ids5 + ids6 + ids5d + ids4d + ids3d + ids2d + ids1d + ids0d

    #         return None, support_points, indices
    #     else:
    #         _, support1, ids1 = self.cv1(None, pos)
    #         _, support2, ids2 = self.cv2(None, support1[0])
    #         _, support3, ids3 = self.cv3(None, support2[0])
    #         _, support4, ids4 = self.cv4(None, support3[0])
    #         _, support5, ids5 = self.cv5(None, support4[0])

    #         support_points = support1 + support2 + support3 + support4 + support5
    #         indices = ids1 + ids2 + ids3 + ids4 + ids5

    #         return None, support_points, indices


    # def forward_with_features(self, x, pos, support_points=None, indices=None):

    #     if self.segmentation:

    #         ids0, ids1, ids2, ids3, ids4, ids5, ids6, ids5d, ids4d, ids3d, ids2d, ids1d, ids0d = indices
    #         support0, support1, support2, support3, support4, support5, support6 = support_points

    #         ids0 = knn(pos, pos, 16, ids0)
    #         x0 = self.activation(self.bn0(self.cv0(x, pos, support0, ids0)))
    #         x1 = self.activation(self.bn1(self.cv1(x0, support0, support1, ids1)))
    #         x2 = self.activation(self.bn2(self.cv2(x1, support1, support2, ids2)))
    #         x3 = self.activation(self.bn3(self.cv3(x2, support2, support3, ids3)))
    #         x4 = self.activation(self.bn4(self.cv4(x3, support3, support4, ids4)))
    #         x5 = self.activation(self.bn5(self.cv5(x4, support4, support5, ids5)))
    #         x6 = self.activation(self.bn6(self.cv6(x5, support5, support6, ids6)))
    #         x = self.activation(self.bn5d(self.cv5d(x6, support6, support5, ids5d)))
    #         x = torch.cat([x, x5], dim=1)
    #         x = self.activation(self.bn4d(self.cv4d(x, support5, support4, ids4d)))
    #         x = torch.cat([x, x4], dim=1)
    #         x = self.activation(self.bn3d(self.cv3d(x, support4, support3, ids3d)))
    #         x = torch.cat([x, x3], dim=1)
    #         x = self.activation(self.bn2d(self.cv2d(x, support3, support2, ids2d)))
    #         x = torch.cat([x, x2], dim=1)
    #         x = self.activation(self.bn1d(self.cv1d(x, support2, support1, ids1d)))
    #         x = torch.cat([x, x1], dim=1)
    #         x = self.activation(self.bn0d(self.cv0d(x, support1, support0, ids0d)))
    #         x = torch.cat([x, x0], dim=1)
    #         x = self.dropout(x)
    #         x = self.fcout(x)

    #     else:

    #         ids1, ids2, ids3, ids4, ids5 = indices
    #         support1, support2, support3, support4, support5 = support_points

    #         x = self.activation(self.bn1(self.cv1(x, pos, support1, ids1)))
    #         x = self.activation(self.bn2(self.cv2(x, support1, support2, ids2)))
    #         x = self.activation(self.bn3(self.cv3(x, support2, support3, ids3)))
    #         x = self.activation(self.bn4(self.cv4(x, support3, support4, ids4)))
    #         x = self.activation(self.bn5(self.cv5(x, support4, support5, ids5)))
    #         x = x.mean(dim=2)
    #         x = self.dropout(x)
    #         x = self.fcout(x)

    #     return x
