from collections import OrderedDict

import torch


def compute_norm(data):
    """Returns the sum of Frobenius norm of elements (torch.tensor) in the
    container `data`.

    Args:
        data(torch.nn.Module, OrderedDict, tuple): input container of tensors.
    """
    if isinstance(data, torch.nn.Module):
        return sum(torch.norm(param) for param in data.parameters())
    elif isinstance(data, OrderedDict):
        return sum(torch.norm(value) for value in data.values())
    elif isinstance(data, tuple):
        return sum(torch.norm(item) for item in data)

