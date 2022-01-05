import torch_geometric.data

class Data(torch_geometric.data.Data):

    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None,
                 pos=None, normal=None, face=None, **kwargs):
        super().__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos, normal=normal, face=face, **kwargs)
        for key, value in kwargs.items():
            self.__dict__[key] = value