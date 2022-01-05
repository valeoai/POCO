import torch
import torch.nn as nn
import lightconvpoint.knn as nearest_neighbors


class UpSampleNearest(nn.Module):
    """Nearest neighbor interpolation.

    # Forward arguments
        input: 3-D torch tensor.
            Input feature tensor. Dimensions are (B, I, N) with B the batch size, I the
            number of input channels and N the number of input points.
        points: 3-D torch tensor.
            The input points. Dimensions are (B, D, N) with B the batch size, D the
            dimension of the spatial space and N the number of input points.
        support_points: 3-D torch tensor.
            The support points to project features on.
            Dimensions are (B, D, N) with B the batch size, D the dimenstion of the
            spatial space and N the number of input points.
        indices: (optional) 3-D torch tensor.
            The indices of the neighboring points with respect to the support points.

    # Forward returns
        features: 3-D torch tensor.
            The computed features. Dimensions are (B, O, N) with B the batch size, O the
            number of output channels and N the number of input points.
        support_points: 3-D torch tensor.
            The support points. If they were provided as an input, return the same
            tensor.
        indices: 3-D torch tensor.
            The indices of the neighboring points with respect to the support points.
            If they were provided as an input, return the same tensor.

    """

    def __init__(self):
        super().__init__()

    def batched_index_select(self, input, dim, index):
        """
        Slicing of input with respect to the index tensor
        """
        index_shape = index.shape
        views = [input.shape[0]] + [
            1 if i != dim else -1 for i in range(1, len(input.shape))
        ]
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index).view(
            input.size(0), -1, index_shape[1], index_shape[2]
        )

    def forward(self, input, points, support_points, indices=None):
        """Forward function of the layer."""

        # support points are known, only compute the knn
        if indices is None:
            indices = nearest_neighbors.knn(
                points.cpu().detach(), support_points.cpu().detach(), 1
            )
            if points.is_cuda:
                indices = indices.cuda()

        if input is None:
            # inpuy is None: do not compute features
            return None, support_points, indices
        else:
            # compute the features
            indices = indices.clone()

            # get the features and point coordinates associated with the indices
            features = self.batched_index_select(
                input, dim=2, index=indices
            ).contiguous()

            return features.squeeze(3), support_points, indices
