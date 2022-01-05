import torch
import math
from torch_geometric.nn.pool import voxel_grid
from lightconvpoint.knn import knn

import importlib
knn_c_func_spec = importlib.util.find_spec('lightconvpoint.knn_c_func')

if knn_c_func_spec is not None:
    knn_c_func = importlib.util.module_from_spec(knn_c_func_spec)
    knn_c_func_spec.loader.exec_module(knn_c_func)

def unique(x, dim=None):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.

    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    unique, inverse = torch.unique(
        x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                        device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return unique, inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)

def sampling_knn_quantized(points: torch.Tensor, nqueries: int, K: int):

    if knn_c_func_spec is not None:
        return knn_c_func.sampling_knn_quantized(points, nqueries, K)

    bs, dim, nx = points.shape

    mini = points.min(dim=2)[0]
    maxi = points.max(dim=2)[0]

    initial_voxel_size = (maxi-mini).norm(2, dim=1) / math.sqrt(nqueries)

    indices_queries = []
    points_queries = []

    for b_id in range(bs):
        voxel_size = initial_voxel_size[b_id]
        x = points[b_id].transpose(0,1)

        b_selected_points = []
        count = 0

        x_ids = torch.arange(x.shape[0])

        while(True):
            batch_x = torch.zeros(x.shape[0], device=points.device, dtype=torch.long)

            voxel_ids = voxel_grid(x,batch_x, voxel_size)
            _, unique_indices = unique(voxel_ids)

            if count + unique_indices.shape[0] >= nqueries:
                unique_indices = unique_indices[torch.randperm(unique_indices.shape[0])]
                b_selected_points.append(x_ids[unique_indices[:nqueries-count]])
                count += unique_indices.shape[0]
                break

            b_selected_points.append(x_ids[unique_indices])
            count += unique_indices.shape[0]

            select = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
            select[unique_indices] = False
            x = x[select]
            x_ids = x_ids[select]
            voxel_size /= 2

        b_selected_points = torch.cat(b_selected_points, dim=0)
        indices_queries.append(b_selected_points)

        points_queries.append(points[b_id].transpose(0,1)[b_selected_points])

    indices_queries = torch.stack(indices_queries, dim=0)
    points_queries = torch.stack(points_queries, dim=0)
    points_queries = points_queries.transpose(1,2)

    indices_knn = knn(points, points_queries, K)

    return indices_queries, indices_knn, points_queries

def sampling_quantized(points: torch.Tensor, nqueries: int):

    if knn_c_func_spec is not None:
        return knn_c_func.sampling_quantized(points, nqueries)

    bs, dim, nx = points.shape

    mini = points.min(dim=2)[0]
    maxi = points.max(dim=2)[0]

    initial_voxel_size = (maxi-mini).norm(2, dim=1) / math.sqrt(nqueries)

    indices_queries = []

    for b_id in range(bs):
        voxel_size = initial_voxel_size[b_id]
        x = points[b_id].transpose(0,1)

        b_selected_points = []
        count = 0

        x_ids = torch.arange(x.shape[0])

        while(True):
            batch_x = torch.zeros(x.shape[0], device=points.device, dtype=torch.long)

            voxel_ids = voxel_grid(x,batch_x, voxel_size)
            _, unique_indices = unique(voxel_ids)

            if count + unique_indices.shape[0] >= nqueries:
                unique_indices = unique_indices[torch.randperm(unique_indices.shape[0])]
                b_selected_points.append(x_ids[unique_indices[:nqueries-count]])
                count += unique_indices.shape[0]
                break

            b_selected_points.append(x_ids[unique_indices])
            count += unique_indices.shape[0]

            select = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
            select[unique_indices] = False
            x = x[select]
            x_ids = x_ids[select]
            voxel_size /= 2

        b_selected_points = torch.cat(b_selected_points, dim=0)
        indices_queries.append(b_selected_points)

    indices_queries = torch.stack(indices_queries, dim=0)

    return indices_queries