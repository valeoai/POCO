import torch
import math
from lightconvpoint.knn import knn

import importlib
knn_c_func_spec = importlib.util.find_spec('lightconvpoint.knn_c_func')

if knn_c_func_spec is not None:
    knn_c_func = importlib.util.module_from_spec(knn_c_func_spec)
    knn_c_func_spec.loader.exec_module(knn_c_func)

def sampling_random(points: torch.Tensor, nqueries: int):

    if knn_c_func_spec is not None:
        return knn_c_func.sampling_random(points, nqueries)

    bs, dim, nx = points.shape

    indices_queries = []

    for b_id in range(bs):

        indices_queries_ = torch.randperm(nx)[:nqueries]
        indices_queries.append(indices_queries_)

    indices_queries = torch.stack(indices_queries, dim=0)

    return indices_queries

def sampling_knn_random(points: torch.Tensor, nqueries: int, K: int):

    if knn_c_func_spec is not None:
        return knn_c_func.sampling_knn_random(points, nqueries, K)

    bs, dim, nx = points.shape

    indices_queries = []
    points_queries = []

    for b_id in range(bs):

        indices_queries_ = torch.randperm(nx)[:nqueries]
        indices_queries.append(indices_queries_)

        x = points[b_id].transpose(0,1)
        points_queries.append(x[indices_queries_])


    indices_queries = torch.stack(indices_queries, dim=0)
    points_queries = torch.stack(points_queries, dim=0)
    points_queries = points_queries.transpose(1,2)

    indices_knn = knn(points, points_queries, K)

    return indices_queries, indices_knn, points_queries