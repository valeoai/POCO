from lightconvpoint.spatial import knn
from lightconvpoint.utils.functional import batch_gather

# def interpolate(x, pos, support_points, neighbors_indices=None, K=1):
#     if neighbors_indices is None:
#         neighbors_indices = knn(pos, support_points, K)
#     if x is None:
#         return None, support_points, neighbors_indices
#     else:
#         x = batch_gather(x, 2, neighbors_indices)
#         if neighbors_indices.shape[-1] > 1:
#             return x.mean(-1), support_points, neighbors_indices
#         else:
#             return x.squeeze(-1), support_points, neighbors_indices

def interpolate(x, neighbors_indices, method="mean"):


    mask = (neighbors_indices > -1)
    neighbors_indices[~mask] = 0

    x = batch_gather(x, 2, neighbors_indices)


    # m = (neighbors_indices[:,:,0] > -1).float().unsqueeze(1).unsqueeze(3)
    # m[m==0] = float("Inf")
    # x = x * m

    if neighbors_indices.shape[-1] > 1:
        if method=="mean":
            return x.mean(-1)
        elif method=="max":
            return x.mean(-1)[0]
    else:
        return x.squeeze(-1)