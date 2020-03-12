import gzip
import os

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

from .distribute import dirichlet_split


class FedFMNIST(Dataset):
    """Federated Fashion-MNIST dataset for a specific client (`client_idx`).
    Note that, if `client_idx` doesn't assigned and `evaluation` is set True,
    it will create a test dataset for clients and the server.

    Args:
        data_dir (string): Directory of dataset exists.
        num_clients (int): Number of clients in federated learning.
        client_idx (optional, int): Index of the client.
        transform (optional, sequence of torchvision.transforms): Common image
        transformations, e.g `transforms.Compose([transforms.ToTensor()])`.
        evaluation: If True, creates federated test dataset, otherwise creates
        fereated training dataset.
    """
    def __init__(self,
                 data_dir,
                 num_clients,
                 client_idx=None,
                 transform=None,
                 evaluation=False):
        if not evaluation and client_idx is None:
            raise TypeError(
                "client_idx doesn't be assigned when evaluation is set True")
        self.classes = 10
        self.transform = transform
        if not evaluation:
            data_file = "train-images-idx3-ubyte.gz"
            targets_file = "train-labels-idx1-ubyte.gz"
        else:
            data_file = "t10k-images-idx3-ubyte.gz"
            targets_file = "t10k-labels-idx1-ubyte.gz"

        data_file_path = os.path.join(data_dir, data_file)
        targets_file_path = os.path.join(data_dir, targets_file)
        with gzip.open(targets_file_path, "rb") as f:
            targets = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        with gzip.open(data_file_path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
            data = data.reshape(len(targets), 28 * 28)

        if not evaluation:
            fmnist_splited = dirichlet_split(data, targets, num_clients,
                                             self.classes)
            self.data, self.targets = fmnist_splited[client_idx]
        # All Clients and the server share the same test data.
        else:
            self.data, self.targets = data, targets

    def __getitem__(self, index):
        data = self.data[index].reshape(28, 28)
        if self.transform is not None:
            data = self.transform(data)
        return data, self.targets[index]

    def __len__(self):
        return len(self.data)
