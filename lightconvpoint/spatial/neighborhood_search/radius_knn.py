import torch
import numpy as np
from scipy.spatial import KDTree

def radius_knn(points, support_points, radius, K, neighbors_indices=None):

    if neighbors_indices is not None:
        return neighbors_indices

    if K > points.shape[2]:
        K = points.shape[2]
    pts = points.cpu().detach().transpose(1,2).numpy().copy()
    s_pts = support_points.cpu().detach().transpose(1,2).numpy().copy()
    n = pts.shape[1]
    indices = []
    for i in range(pts.shape[0]):
        tree = KDTree(pts[i])
        
        indices_batch = []
        for results in tree.query_ball_point(s_pts[i], r=radius):
            results = torch.tensor(results, dtype=torch.long)
            if results.shape[0] > K:
                results = results[torch.randperm(results.shape[0])[:K]]
            else:
                ids = torch.arange(results.shape[0], dtype=torch.long)
                ids = ids.repeat(K//ids.shape[0]+1)
                ids = ids[torch.randperm(ids.shape[0])[:K]]
                results = results[ids]
            indices_batch.append(results)
        indices_batch = torch.stack(indices_batch, dim=0)
        indices.append(indices_batch)
    indices = torch.stack(indices, dim=0)
    if K==1:
        indices = indices.unsqueeze(2)
    return indices.to(points.device)