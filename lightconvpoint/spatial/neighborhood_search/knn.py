import torch
from scipy.spatial import KDTree

def knn(points, support_points, K, neighbors_indices=None):

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
        _, indices_ = tree.query(s_pts[i], k=K)
        indices.append(torch.tensor(indices_, dtype=torch.long))
    indices = torch.stack(indices, dim=0)
    if K==1:
        indices = indices.unsqueeze(2)
    return indices.to(points.device)