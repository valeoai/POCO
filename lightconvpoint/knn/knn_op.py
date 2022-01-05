import torch
from torch_geometric.nn.pool import knn as tc_knn

import importlib
knn_c_func_spec = importlib.util.find_spec('lightconvpoint.knn_c_func')

if knn_c_func_spec is not None:
    knn_c_func = importlib.util.module_from_spec(knn_c_func_spec)
    knn_c_func_spec.loader.exec_module(knn_c_func)

def knn(points: torch.Tensor, queries: torch.Tensor, K: int):

    if knn_c_func_spec is not None:
        return knn_c_func.knn(points, queries, K)

    bs = points.shape[0]
    dim= points.shape[1]
    nx = points.shape[2]
    ny = queries.shape[2]

    K = min(K, nx)

    batch_x = torch.arange(0, bs, dtype=torch.long, device=points.device).unsqueeze(1).expand(bs,nx)
    batch_y = torch.arange(0, bs, dtype=torch.long, device=queries.device).unsqueeze(1).expand(bs,ny)

    x = points.transpose(1,2).reshape(-1, dim)
    y = queries.transpose(1,2).reshape(-1, dim)
    batch_x = batch_x.view(-1)
    batch_y = batch_y.view(-1)

    indices = tc_knn(x,y,K,batch_x=batch_x, batch_y=batch_y)
    indices = indices[1]
    return indices.view(bs,ny,K)

