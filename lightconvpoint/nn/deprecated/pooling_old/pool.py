import torch

def batched_index_select(input, dim, index):
    """Gather input with respect to the index tensor."""
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


def max_pool(input, indices):
    """Forward function of the layer."""
    features = batched_index_select(input, dim=2, index=indices).contiguous()
    features = features.max(dim=3)[0]
    return features

def mean_pool(input, indices):
    """Forward function of the layer."""
    features = batched_index_select(input, dim=2, index=indices).contiguous()
    features = features.mean(dim=3)
    return features

def min_pool(input, indices):
    """Forward function of the layer."""
    features = batched_index_select(input, dim=2, index=indices).contiguous()
    features = features.min(dim=3)[0]
    return features