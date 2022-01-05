import torch
import math
from torch_geometric.nn.pool import fps
from lcp.knn import knn

import importlib
knn_c_func_spec = importlib.util.find_spec('lightconvpoint.knn_c_func')

if knn_c_func_spec is not None:
    knn_c_func = importlib.util.module_from_spec(knn_c_func_spec)
    knn_c_func_spec.loader.exec_module(knn_c_func)

def sampling_fps(points: torch.Tensor, nqueries: int):

    if knn_c_func_spec is not None:
        return knn_c_func.sampling_fps(points, nqueries)

    bs, dim, nx = points.shape

    ratio = nqueries / nx

    batch_x = torch.arange(0, bs, dtype=torch.long, device=points.device).unsqueeze(1).expand(bs,nx)

    x = points.transpose(1,2).reshape(-1, dim)
    batch_x = batch_x.view(-1)

    indices_queries = fps(x, batch_x, ratio)

    indices_queries = indices_queries.view(bs, -1)

    assert(indices_queries.shape[1] == nqueries)
    return indices_queries



def sampling_knn_fps(points: torch.Tensor, nqueries: int, K: int):

    if knn_c_func_spec is not None:
        return knn_c_func.sampling_knn_fps(points, nqueries, K)

    bs, dim, nx = points.shape

    ratio = nqueries / nx

    batch_x = torch.arange(0, bs, dtype=torch.long, device=points.device).unsqueeze(1).expand(bs,nx)

    x = points.transpose(1,2).reshape(-1, dim)
    batch_x = batch_x.view(-1)

    indices_queries = fps(x, batch_x, ratio)

    points_queries = x[indices_queries]

    indices_queries = indices_queries.view(bs, -1)
    points_queries = points_queries.view(bs,-1,3)
    points_queries = points_queries.transpose(1,2)

    assert(indices_queries.shape[1] == nqueries)

    indices_knn = knn(points, points_queries, K)

    return indices_queries, indices_knn, points_queries