import torch
from lightconvpoint.utils.functional import batch_gather

def sampling_random(pos, ratio=None, n_support=None, support_points=None, support_points_ids=None):

    if support_points is not None:
        return support_points, support_points_ids

    assert((ratio is None) != (n_support is None))


    if ratio is not None:
        support_point_number = max(1,int(pos.shape[2] * ratio))
    else:
        support_point_number = n_support

    if support_point_number == pos.shape[2]:
        support_points_ids = torch.arange(pos.shape[2], dtype=torch.long, device=pos.device)
        support_points_ids = support_points_ids.unsqueeze(0).expand(pos.shape[0], pos.shape[2])
        return pos, support_points_ids
    elif support_point_number>0 and support_point_number<pos.shape[2]:

        # if replacement:
        #     support_points_ids = torch.rand((points.shape[0], support_point_number)) * points.shape[2]
        #     support_points_ids = support_points_ids.long()
        # else:

        support_points_ids = []
        for i in range(pos.shape[0]):
            support_points_ids.append(torch.randperm(pos.shape[2])[:support_point_number])
        support_points_ids = torch.stack(support_points_ids, dim=0)
        
        support_points_ids = support_points_ids.to(pos.device)
        support_points = batch_gather(pos, dim=2, index=support_points_ids)
        return support_points, support_points_ids
    else:
        raise ValueError(f"Search random - ratio value error {ratio} should be in ]0,1]")
