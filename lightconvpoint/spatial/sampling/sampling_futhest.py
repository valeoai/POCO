import torch
import logging
from lightconvpoint.utils.functional import batch_gather

def sampling_furthest(points, ratio=1, return_support_points=False):
    if ratio==1:
        support_points_ids = torch.arange(points.shape[2], dtype=torch.long, device=points.device)
        support_points_ids = support_points_ids.unsqueeze(0).expand(points.shape[0], points.shape[2])
        if return_support_points:
            return support_points_ids, points
    elif ratio>0 and ratio<1:
        
        raise NotImplementedError
        
        support_points_ids = support_points_ids.to(points.device)
        if return_support_points:
            support_points = batch_gather(points, dim=2, index=support_points_ids)
            return support_points_ids, support_points
        else:
            return support_points_ids
    else:
        raise ValueError(f"Search FPS - ratio value error {ratio} should be in ]0,1]")