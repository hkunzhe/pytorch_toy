import pickle
import os

import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset


class CIFAR10(Dataset):
    """CIFAR10 dataset.

    Args:
        data_dir (string): Directory of dataset exists.
        idx (int): The `idx`-th sub-data of CIFAR10. 
        indices (list): The indices to split CIFAR10.
        transform (callable, optional): A function/transform that takes in an
            PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        evaluation (bool, optional): If True, creates test dataset, otherwise creates training
            dataset.
    """
    def __init__(self,
                 data_dir,
                 idx=None,
                 indices=[30000, 40000, 50000],
                 transform=None,
                 backdoor_transform=None,
                 evaluation=False):
        self.classes = 10
        self.transform = transform
        self.backdoor_transform = backdoor_transform
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

        # Convert data (List) to NHWC (np.ndarray) works with transformations.
        data = np.vstack(data).reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))

        if not evaluation:
            # Split and fetch idx-th data and target.
            self.data = np.vsplit(data, indices)[idx]
            self.targets = np.split(np.asarray(targets), indices)[idx]
        else:
            self.data = data
            self.targets = np.asarray(targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # Do the backdoor transformation first and then normal transformations.
        if self.backdoor_transform is not None:
            img, target = self.backdoor_transform(img, target)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)
