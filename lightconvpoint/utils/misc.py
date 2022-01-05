import torch

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# wrap blue / green
def wblue(str):
    return bcolors.OKBLUE+str+bcolors.ENDC
def wgreen(str):
    return bcolors.OKGREEN+str+bcolors.ENDC
def wred(str):
    return bcolors.FAIL+str+bcolors.ENDC



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def list_to_device(data, device):
    for key, value in enumerate(data):
        if torch.is_tensor(value):
            data[key] = value.to(device)
        elif isinstance(value, list):
            data[key] = list_to_device(value, device)
        elif isinstance(value, dict):
            data[key] = dict_to_device(value, device)
    return data

def dict_to_device(data, device):
    for key, value in data.items():
        if torch.is_tensor(value):
            data[key] = value.to(device)
        elif isinstance(value, list):
            data[key] = list_to_device(value, device)
        elif isinstance(value, dict):
            data[key] = dict_to_device(value, device)
    return data
