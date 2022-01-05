import torch
from torch.utils.data._utils.collate import default_collate
from torch_geometric.data import Data


def collate(data_list):

    batch = []
    for data in data_list:
        d = {}
        for key in data.keys:
            d[key] = data[key]
        batch.append(d)

    batch = default_collate(batch)

    return Data(**batch)


def get_dataset(base_class):

    # create a dataset class that will inherit from base_class
    class LCPDataset(base_class):

        def __init__(self, *args, **kwargs):

            if "network_function" in kwargs:
                net_func = kwargs["network_function"]
                del kwargs["network_function"]
            else:
                net_func = None

            super().__init__(*args, **kwargs)

            if net_func is not None:
                self.net = net_func()
            else:
                self.net = None


        def download(self):
            super().download()

        def process(self):
            super().process()

        def __getitem__(self, idx):

            data = super().__getitem__(idx)

            if (self.net is not None) and ("lcp_preprocess" in self.net.__dict__) and (self.net.__dict__["lcp_preprocess"]):

                with torch.no_grad():
                    return_data = self.net(data, spatial_only=True)

                for key in return_data.keys():
                    if return_data[key] is not None:
                        if isinstance(return_data[key], torch.Tensor):
                            data[key] = return_data[key].detach()
                        else:
                            data[key] = return_data[key]

            # remove None type keys
            to_delete_keys = []
            for key in data:
                if data[key] is None:
                    to_delete_keys.append(key)

            for key in to_delete_keys:
                data.pop(key, None)
            
            return data

    return LCPDataset