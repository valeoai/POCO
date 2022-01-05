from operator import index
from scipy.spatial import KDTree
import torch
import numpy as np

def radius(points, support_points, radius, max_num_neighbors, neighbors_indices=None):

    # if max num_neighbors defines the maximum number of neighbors
    # and the size of the out vector

    if neighbors_indices is not None:
        return neighbors_indices

    # move to cpu numpy
    pts = points.cpu().detach().transpose(1,2).numpy().copy()
    s_pts = support_points.cpu().detach().transpose(1,2).numpy().copy()
    n = pts.shape[1]
    indices = []

    # iterate in the batch dimenstion
    for i in range(pts.shape[0]):

        # build the KDTree
        tree_pts = KDTree(pts[i])
        tree_support = KDTree(s_pts[i])

        indices_query = tree_support.query_ball_tree(tree_pts, r=radius)

        # create the indices matrix
        indices_ = torch.full((s_pts.shape[1], max_num_neighbors), -1, dtype=torch.long)

        for i in range(len(indices_query)):
            
            ids = torch.tensor(indices_query[i], dtype=torch.long)

            if ids.shape[0]==0: # no neighbors
                continue

            if ids.shape[0] > max_num_neighbors: # use all neighbors
                ids = ids[torch.randperm(ids.shape[0])][:max_num_neighbors]

            indices_[i, :ids.shape[0]] = ids
        
        indices.append(indices_)

    indices = torch.stack(indices, dim=0)

    if max_num_neighbors == 1:
        indices = indices.unsqueeze(2)

    return indices.to(points.device)