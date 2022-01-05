import torch
import torch.nn as nn

class Module(nn.Module):

    def __init__(self):
        super().__init__()

    def forward_without_features(self, pos, support_points=None, indices=None):
        raise NotImplementedError

    def forward_with_features(self, x, pos, support_points, indices):
        raise NotImplementedError

    def forward(self, x, pos, support_points=None, indices=None):
        if x is None:
            return self.forward_without_features(pos, support_points, indices)
        else:
            if (support_points is None) or (indices is None):
                _, support_points, indices = self.forward_without_features(pos, support_points, indices)
            if isinstance(support_points, torch.Tensor):
                support_points = [support_points]
            if isinstance(indices, torch.Tensor):
                indices = [indices]
            return self.forward_with_features(x, pos, support_points, indices)
