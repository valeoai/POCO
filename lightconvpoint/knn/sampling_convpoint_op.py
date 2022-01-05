import torch
import math
from torch_geometric.nn.pool import voxel_grid
from lcp.knn import knn

import importlib
knn_c_func_spec = importlib.util.find_spec('lightconvpoint.knn_c_func')

if knn_c_func_spec is not None:
    knn_c_func = importlib.util.module_from_spec(knn_c_func_spec)
    knn_c_func_spec.loader.exec_module(knn_c_func)


def sampling_knn_convpoint(points: torch.Tensor, nqueries: int, K: int):

    if knn_c_func_spec is not None:
        return knn_c_func.sampling_knn_convpoint(points, nqueries, K)
    else:
        raise NotImplementedError
