import torch
import logging
from lightconvpoint.utils.functional import batch_gather

def sampling_convpoint(pos, ratio=1, support_points=None, support_points_ids=None, K=16):

    if ratio == 1:
        support_points_ids = torch.arange(pos.shape[2], dtype=torch.long, device=pos.device)
        support_points_ids = support_points_ids.unsqueeze(0).expand(pos.shape[0], pos.shape[2])
        return pos, support_points_ids
    elif ratio>0 and ratio<1:
        
        raise NotImplementedError
        
        support_points_ids = support_points_ids.to(pos.device)
        support_points = batch_gather(pos, dim=2, index=support_points_ids)
        return support_points, support_points_ids

    else:
        raise ValueError(f"Search ConvPoint - ratio value error {ratio} should be in ]0,1]")