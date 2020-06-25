import os
import pickle
import random

import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset


class CIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None, **kwargs):
        self.transform = transform
        if train:
            data_list = [
                "data_batch_1",
                "data_batch_2",
                "data_batch_3",
                "data_batch_4",
                "data_batch_5",
            ]
        else:
            data_list = ["test_batch"]

        data = []
        targets = []
        for file_name in data_list:
            file_path = os.path.join(root, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
            data.append(entry["data"])
            targets.extend(entry["labels"])
        # Convert data (List) to NHWC (np.ndarray) works with transformations.
        data = np.vstack(data).reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
        self.data = data
        self.targets = np.asarray(targets)

        self.kwargs = kwargs
        self.pidx = [0] * len(data)  # poisoned index (0/1 stands for clean/poisoned)
        if len(kwargs) != 0:
            assert "tlabel" in kwargs
            assert "pratio" in kwargs
            assert "bd_transform" in kwargs
            for (i, t) in enumerate(targets):
                if random.random() < kwargs["pratio"] and t != kwargs["tlabel"]:
                    self.pidx[i] = 1

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        poisoned = 0
        # Do the backdoor transformation first and then normal transformations.
        if self.pidx[index] == 1:
            img = self.kwargs["bd_transform"](img)
            target = self.kwargs["tlabel"]
            poisoned = 1
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target, poisoned

    def __len__(self):
        return len(self.data)
