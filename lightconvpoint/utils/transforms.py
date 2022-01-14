from os import replace
import numpy as np
from torchvision import transforms
from PIL import Image
import random
import logging
import torch
import re
import math
import numbers
from scipy.spatial import ConvexHull

class RandomNoiseNormal(object):
    
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, data):

        data["pos"] += self.sigma * torch.randn_like(data["pos"])
        if "normal" in data:
            data["normal"] += self.sigma * torch.randn_like(data["normal"])
            data["normal"] /= data["normal"].norm(dim=-1, keepdim=True)
        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.sigma)


class SubSamplePad(object):

    def __init__(self, npoints) -> None:
        super().__init__()
        self.npoints = npoints

    def __call__(self, data):

        num_nodes = data["pos"].shape[0]

        if num_nodes > self.npoints: # subsample
            choice = torch.randperm(num_nodes)[:self.npoints]
            for key, item in data:
                if bool(re.search('edge', key)):
                    continue
                if (torch.is_tensor(item) and item.size(0) == num_nodes
                        and item.size(0) != 1):
                    data[key] = item[choice]
        else: # pad
            # print("PAD")
            for key, item in data:
                if bool(re.search('edge', key)):
                    continue
                if (torch.is_tensor(item) and item.size(0) == num_nodes and item.size(0) != 1):
                    pad_size = (self.npoints - num_nodes,) + item.shape[1:]
                    if item.dtype == torch.long:
                        data[key] = torch.cat([item, torch.full(pad_size, -1, dtype=torch.long)], dim=0)
                    else:
                        data[key] = torch.cat([item, torch.full(pad_size, float("Inf") )], dim=0)

        return data


class RandomPillarSelection(object):

    def __init__(self, pillar_size, infinite_dim=2):
        self.pillar_size = pillar_size
        self.infinite_dim = infinite_dim

    def __call__(self, data):

        npoints = data["pos"].shape[0]

        # find a center
        pt_id = torch.randint(0,npoints,size=(1,)).item()
        pillar_center = data["pos"][pt_id]
        pillar_center[2] = 0

        # compute the mask
        mask = None
        for i in range(pillar_center.shape[0]):
            if self.infinite_dim != i:
                mask_i = torch.logical_and(
                            data["pos"][:,i]<=pillar_center[i]+self.pillar_size/2,
                            data["pos"][:,i]>=pillar_center[i]-self.pillar_size/2)
                if mask is None:
                    mask = mask_i
                else:
                    mask = torch.logical_and(mask, mask_i)

        # apply the mask
        for key, value in data.__dict__.items():
            if isinstance(value, torch.Tensor) and value.shape[0]==npoints:
                data[key] = value[mask]

        data["pos"] = data["pos"] - pillar_center

        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.pillar_size, self.infinite_dim)


def uniform_point_selector(data):
    npoints = data["pos"].shape[0]
    pt_id = torch.randint(0, npoints, size=(1,)).item()
    return pt_id

def distance_point_selector(data):
    distances = data["pos"].norm(dim=1)
    distances /= distances.sum()
    distances = torch.cumsum(distances, dim=0)
    prob = torch.rand(size=(1,)).item()
    pt_id = torch.abs(distances-prob).argmin()
    return pt_id

class RandomBallSelection(object):

    def __init__(self, radius, center_selection_function=distance_point_selector):
        self.radius = radius
        self.selection_function = center_selection_function

    def __call__(self, data):

        npoints = data["pos"].shape[0]

        if npoints > 1:
            # find a center 
            
            count=0
            while count<100:
                pt_id = self.selection_function(data)
                ball_center = data["pos"][pt_id]
                mask = ((data["pos"] - ball_center.unsqueeze(0)).norm(dim=1) < self.radius)
                count +=1
                if mask.sum()>1:
                    break

            if count >= 100:
                raise ValueError("Reached number of sample for ball selection")

            # apply the mask
            for key, value in data.__dict__.items():
                if isinstance(value, torch.Tensor) and value.shape[0]==npoints:
                    data[key] = value[mask]

            return data

    def __repr__(self):
        return f'{self.__class__.__name__}({self.radius})'


class UnitBallNormalize(object):
    def __init__(self, item_list=["pos"], multiplier=1):
        self.item_list = item_list
        self.multiplier = multiplier

    def __call__(self, data):  # from KPConv code

        pts = data["pos"].clone()
        pmin = pts.min(dim=0)[0]
        pmax = pts.max(dim=0)[0]
        translation = (pmin + pmax) / 2
        pts -= translation
        scale = pts.norm(dim=1).max()

        # print("pts", pts.shape)
        # print(translation)
        # print(scale)

        for key, item in data:
            if key in self.item_list:
                if torch.is_tensor(item):
                    data[key] = data[key] - translation
                    data[key] = data[key] / scale
                    data[key] = data[key] * self.multiplier

                    


        data["normalization_translation"] = translation
        data["normalization_scale"] = scale
        data["normalization_multiplier"] = self.multiplier

        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__)


class ColorJittering(object):

    def __init__(self, jitter_value):
        self.jiter_value = jitter_value
        self.transform = transforms.ColorJitter(
            brightness=jitter_value,
            contrast=jitter_value,
            saturation=jitter_value)


    def __call__(self, data):

        x = (data["x"] * 255).cpu().numpy().astype(np.uint8)
        x = np.array(self.transform( Image.fromarray(np.expand_dims(x, 0))))
        x = np.squeeze(x, 0)
        x = torch.tensor(x, device=data["x"].device, dtype=data["x"].dtype)/255
        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.jitter_value)

class FixedPoints(object):

    # extension of the Fixed points from torch_geometric with a given list of item to sample

    def __init__(self, num, replace=True, allow_duplicates=False, item_list=None):
        self.num = num
        self.replace = replace
        self.allow_duplicates = allow_duplicates
        self.item_list = item_list

    def __call__(self, data):
        if self.item_list is None:
            num_nodes = data.num_nodes
        else:
            num_nodes = data[self.item_list[0]].shape[0]

        if self.replace:
            choice = np.random.choice(num_nodes, self.num, replace=True)
            choice = torch.from_numpy(choice).to(torch.long)
        elif not self.allow_duplicates:
            choice = torch.randperm(num_nodes)[:self.num]
        else:
            choice = torch.cat([
                torch.randperm(num_nodes)
                for _ in range(math.ceil(self.num / num_nodes))
            ], dim=0)[:self.num]

        if self.item_list is None:
            for key, item in data:
                if bool(re.search('edge', key)):
                    continue
                if (torch.is_tensor(item) and item.size(0) == num_nodes
                        and item.size(0) != 1):
                    data[key] = item[choice]
        else:
            for key, item in data:
                if key in self.item_list:
                    if bool(re.search('edge', key)):
                        continue
                    if (torch.is_tensor(item) and item.size(0) != 1):
                        data[key] = item[choice]
        return data

    def __repr__(self):
        return '{}({}, replace={})'.format(self.__class__.__name__, self.num,
                                           self.replace)


class Permutation(object):

    # extension of the Fixed points from torch_geometric with a given list of item to sample

    def __init__(self, key, permute=None):
        self.key = key
        self.permute = permute

    def __call__(self, data):

        if self.key in data.keys:
            data[self.key] = data[self.key].permute(self.permute)
        return data

    def __repr__(self):
        return '{}({}, permute={})'.format(self.__class__.__name__, self.key,
                                           self.permute)


class ToDict(object):

    def __call__(self, data):

        d = {}
        for key in data.keys:
            d[key] = data[key]
        return d

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)



class FixedNormalization(object):
    def __init__(self, scale, remove_mean=False, item_list=["pos"]):
        self.scale = scale
        self.item_list = item_list
        self.remove_mean = remove_mean

    def __call__(self, data):

        for key, item in data:
            if key in self.item_list:
                if torch.is_tensor(item):
                    data[key] = item * self.scale
                    if self.remove_mean:
                        data[key] = data[key] - data[key].mean(0)[None,:]
        return data

class RandomRotate(object):
    def __init__(self, degrees, axis=0, item_list=["pos"]):
        if isinstance(degrees, numbers.Number):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        self.degrees = degrees
        self.axis = axis
        self.item_list = item_list

    def __call__(self, data):
        degree = math.pi * random.uniform(*self.degrees) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)

        if data.pos.size(-1) == 2:
            matrix = [[cos, sin], [-sin, cos]]
        else:
            if self.axis == 0:
                matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
            elif self.axis == 1:
                matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
            else:
                matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]

        matrix = torch.tensor(matrix)


        for key, item in data:
            if key in self.item_list:
                if torch.is_tensor(item):
                    data[key] = torch.matmul(item, matrix.to(item.dtype).to(item.device))

        return data



        return LinearTransformation(torch.tensor(matrix))(data)

    def __repr__(self):
        return '{}({}, axis={})'.format(self.__class__.__name__, self.degrees,
                                        self.axis)


class TransposeChannels(object):

    # extension of the Fixed points from torch_geometric with a given list of item to sample

    def __init__(self, item_list=["x", "pos"], dimA=0, dimB=1):
        self.dimA = dimA
        self.dimB = dimB
        self.item_list = item_list

    def __call__(self, data):

        for key in self.item_list:
            data[key] = data[key].transpose(self.dimA, self.dimB)

        return data

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class FieldAsFeatures(object):

    def __init__(self, item_list=None):
        self.item_list = item_list

    def __call__(self, data):
        features = []
        for key in self.item_list:
            features.append(data[key])
        features = torch.cat(features, dim=1)
        data["x"] = features
        return data

class Unsqueeze(object):

    # extension of the Fixed points from torch_geometric with a given list of item to sample

    def __init__(self, item_list=["x", "pos"], dim=0):
        self.dim = dim
        self.item_list = item_list

    def __call__(self, data):

        for key in self.item_list:
            data[key] = data[key].unsqueeze(self.dim)

        return data

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)

class PartialView(object):
    def __init__(self, item_list=None, replace_features_with_dirs=False, view_point_num=1):
        self.item_list = item_list
        self.replace_features = replace_features_with_dirs
        self.view_point_num = view_point_num

    def __call__(self, data):


        vertex_ids = []
        directions = []
        for i in range(self.view_point_num):
            ### from Direct Visibility of Point Sets, Katz et al.
            # pick a random view point
            angles = torch.rand(size=(2,)) * 2 * math.pi
            u = torch.cos(angles[1])
            x_view = torch.sqrt(1-u) * torch.cos(angles[0])
            y_view = torch.sqrt(1-u) * torch.sin(angles[0])
            z_view = u
            view_point = torch.tensor([x_view, y_view, z_view])
            view_point = view_point * 2 # view point outside the enclosing ball

            # set the origin to view point
            data_shift = data["pos"]-view_point
            data_shift_norm = data_shift.norm(2, dim=1).unsqueeze(1)

            # data shift
            R_param = 2
            R = data_shift_norm.max(dim=0)[0] * 10**R_param
            data_shift_mirror = data_shift + 2*(R - data_shift_norm) * data_shift / data_shift_norm

            # Convex hull
            points = np.concatenate([[[0,0,0]], data_shift_mirror.numpy()])
            hull = ConvexHull(points)
            v_ids = torch.from_numpy(hull.vertices[hull.vertices>0] -1).long()

            vertex_ids.append(v_ids)
            directions.append((- torch.nn.functional.normalize(data["pos"][v_ids]-view_point, dim=1)))

        vertex_ids = torch.cat(vertex_ids, dim=0)
        directions = torch.cat(directions, dim=0)


        for key, item in data:
            if key in self.item_list:
                if bool(re.search('edge', key)):
                    continue
                if (torch.is_tensor(item) and item.size(0) != 1):
                    data[key] = item[vertex_ids]

        if self.replace_features:
            data["x"] = directions
        
        return data


# class PillarSelection:

#     def __init__(self, pillar_size, infinite_pillar_dim=2):
#         logging.warning("Deprecation of this data augmentation class")
#         self.pillar_size = pillar_size
#         self.infinite_pillar_dim = infinite_pillar_dim

#     def __call__(self, data, pillar_center=None, **kwargs):

#         # data should have shape x,y,z,...
#         if pillar_center is None:
#             pillar_center = data[random.randint(0, data.shape[0]-1), :3]

#         # compute the mask
#         mask = None
#         for i in range(pillar_center.shape[0]):
#             if self.infinite_pillar_dim != i:
#                 mask_i = np.logical_and(data[:,i]<=pillar_center[i]+self.pillar_size/2,
#                                         data[:,i]>=pillar_center[i]-self.pillar_size/2)
#                 if mask is None:
#                     mask = mask_i
#                 else:
#                     mask = np.logical_and(mask, mask_i)

#         # apply the mask
#         return data[mask]

# class BallSelection:

#     def __init__(self, radius):
#         logging.warning("Deprecation of this data augmentation class")
#         self.radius = radius

#     def __call__(self, data, ball_center=None, return_mask=False, **kwargs):

#         # data should have shape x,y,z,...
#         if ball_center is None:
#             ball_center = data[random.randint(0, data.shape[0]-1), :3]

#         distances = np.linalg.norm(data[:,:3] - ball_center[None,:], axis=1)
#         mask = distances < self.radius

#         # apply the mask
#         if return_mask:
#             return data[mask], mask
#         else:
#             return data[mask]


# class RandomSubSample:

#     def __init__(self, number_of_points):
#         logging.warning("Deprecation of this data augmentation class")
#         self.n = number_of_points

#     def __call__(self, data, return_choice=False, **kwargs):
#         choice = np.random.choice(data.shape[0], self.n, replace=(data.shape[0] < self.n))
#         if return_choice:
#             return data[choice], choice
#         else:
#             return data[choice]


# class FixedSubSample:

#     # subsample by taking the k first points
#     # this is no random

#     def __init__(self, number_of_points):
#         logging.warning("Deprecation of this data augmentation class")
#         self.n = number_of_points

#     def __call__(self, data, return_choice=False, **kwargs):
#         if return_choice:
#             return data[:self.n], np.arange(self.n)
#         else:
#             return data[:self.n]


# class RandomRotation:

#     def __init__(self, rotation_axis=2):
#         logging.warning("Deprecation of this data augmentation class")
#         self.rotation_axis=rotation_axis

#     def __call__(self, points, **kwargs):

#         rotation_angle = np.random.uniform() * 2 * np.pi
#         cosval = np.cos(rotation_angle)
#         sinval = np.sin(rotation_angle)

#         if self.rotation_axis==2:
#             rotation_matrix = np.array([[cosval, sinval, 0],
#                                         [-sinval, cosval, 0],
#                                         [0, 0, 1],])
#         elif self.rotation_axis==1:
#             rotation_matrix = np.array([[cosval, 0, sinval],
#                                         [0, 1, 0]
#                                         [-sinval, 0, cosval],])
#         elif self.rotation_axis==0:
#             rotation_matrix = np.array([[1, 0, 0],
#                                         [0, cosval, sinval],
#                                         [0, -sinval, cosval],])
#         else:
#             raise ValueError("Bad rotation axis")

#         return points @ rotation_matrix




# class ColorDropout:

#     def __init__(self, dropout_value):
#         logging.warning("Deprecation of this data augmentation class")
#         self.dropout_value = dropout_value
    
#     def __call__(self, features, **kwargs):

#         if np.random.rand() < self.dropout_value:
#             return np.ones_like(features)
#         return features


# class NoColor:

#     def __init__(self) -> None:
#         logging.warning("Deprecation of this data augmentation class")

#     def __call__(self, features, **kwargs):
#         return np.ones_like(features)