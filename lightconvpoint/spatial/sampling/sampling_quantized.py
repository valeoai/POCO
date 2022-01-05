import torch
import math
from torch_geometric.data import Data
from torch_geometric.nn import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from lightconvpoint.utils.functional import batch_gather
import logging
from torch_geometric.transforms import RandomRotate

def sampling_quantized(pos, ratio=None, n_support=None, support_points=None, support_points_ids=None):


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

        # voxel_size
        maxi, _ = torch.max(pos, dim=2)
        mini, _ = torch.min(pos, dim=2)
        vox_size = (maxi-mini).norm(2, dim=1)/ math.sqrt(support_point_number)

        Rx = RandomRotate(180, axis=0)
        Ry = RandomRotate(180, axis=1)
        Rz = RandomRotate(180, axis=2)

        support_points_ids = []
        for i in range(pos.shape[0]):
            pts = pos[i].clone().transpose(0,1)
            ids = torch.arange(pts.shape[0])
            sampled_count = 0
            sampled = []
            vox = vox_size[i]
            while(True):
                data = Data(pos=pts)
                data = Rz(Ry(Rx(data)))

                c = voxel_grid(data.pos, torch.zeros(data.pos.shape[0]), vox)
                _, perm = consecutive_cluster(c)

                if sampled_count + perm.shape[0] < support_point_number:
                    sampled.append(ids[perm])
                    sampled_count += perm.shape[0]

                    tmp = torch.ones_like(ids)
                    tmp[perm] = 0
                    tmp = (tmp > 0)
                    pts = pts[tmp]
                    ids = ids[tmp]
                    vox = vox / 2
                    # pts = pts[perm]
                    # ids = ids[perm]
                else:
                    n_to_select = support_point_number - sampled_count
                    perm = perm[torch.randperm(perm.shape[0])[:n_to_select]]
                    sampled.append(ids[perm])
                    break
            sampled = torch.cat(sampled)
            support_points_ids.append(sampled)

        support_points_ids = torch.stack(support_points_ids, dim=0)


        support_points_ids = support_points_ids.to(pos.device)

        support_points = batch_gather(pos, dim=2, index=support_points_ids)
        return support_points, support_points_ids
    else:
        raise ValueError(f"Search Quantized - ratio value error {ratio} should be in ]0,1]")