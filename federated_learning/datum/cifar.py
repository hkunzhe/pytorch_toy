import pickle
import os

import numpy as np
from torch.utils.data.dataset import Dataset

from .distribute import dirichlet_split


class FedCIFAR10(Dataset):
    """Federated CIFAR10 dataset for a specific client (`client_idx`).
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
            data_list = [
                "data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4",
                "data_batch_5"
            ]
        else:
            data_list = ["test_batch"]

        data = []
        targets = []
        for file_name in data_list:
            file_path = os.path.join(data_dir, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
            data.append(entry["data"])
            targets.extend(entry["labels"])

        data = np.vstack(data).reshape(-1, 3, 32, 32)
        data = data.transpose((0, 2, 3, 1))  # convert to HWC
        targets = np.asarray(targets)

        if not evaluation:
            cifar_splited = dirichlet_split(data, targets, num_clients,
                                            self.classes)
            self.data, self.targets = cifar_splited[client_idx]
        # All Clients and the server share the same test data.
        else:
            self.data, self.targets = data, targets

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)
