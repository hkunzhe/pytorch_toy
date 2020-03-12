import copy
from collections import defaultdict, OrderedDict

import numpy as np
import torch


def dirichlet_split(data, targets, num_clients, num_classes, alpha=0.9):
    """Split each class of `data` by Dirichlet Sampling to obtain a
    federated non-i.i.d dataset for clients.

    Args:
        data (np.ndarray): [num_samples, data_point].
        targets (np.ndarray): corresponding targets for data.
        num_clients: number of clients in federated learning.
        num_classes: number of class in `data`.
        alpha (int): parameter of Dirichlet distribution.

    Returns:
        defaultdict: The key (int) is the index of every client, and
        the value (tuple(list, list)) is the splited data with targets
        of the corresponding client.
    """
    np.random.seed(100)
    data_splited = defaultdict(lambda: ([], []))
    class_size = len(data) / num_classes
    sample_prob = np.random.dirichlet(np.array(num_clients * [alpha]))

    for class_ in range(num_classes):
        # filter a specific class in data
        data_by_class = copy.deepcopy(
            data[np.argwhere(targets == class_).squeeze()])
        np.random.shuffle(data_by_class)

        # distribute data with the same class to clients
        i = 0
        for client in range(num_clients):
            num_samples = int(round(class_size * sample_prob[client]))
            j = i + num_samples if client != num_clients - 1 else len(
                data_by_class)
            sample_data = data_by_class[i:j]
            i = j
            sample_targets = np.array([class_] * len(sample_data))
            data_splited[client][0].extend(sample_data)
            data_splited[client][1].extend(sample_targets)

    return data_splited
