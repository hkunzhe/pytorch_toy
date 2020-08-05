import os
import pickle
from copy import deepcopy

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import Dataset

from .backdoor import AddTrigger


class CIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None):
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

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        poisoned = 0
        origin = target  # original target
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target, poisoned, origin

    def __len__(self):
        return len(self.data)

    def set_transform(self, crop=None, hflip=None, vflip=None, erase=None):
        transform = []
        if crop is not None:
            copied_crop = deepcopy(crop)
            size = copied_crop.pop("size")
            transform.append(transforms.RandomCrop(size, **copied_crop))
        if hflip is not None:
            transform.append(transforms.RandomHorizontalFlip(hflip["p"]))
        if vflip is not None:
            transform.append(transforms.RandomVerticalFlip(vflip["p"]))
        transform.append(transforms.ToTensor())
        transform.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        )
        if erase is not None:
            transform.append(transforms.RandomErasing(**erase))
        self.transform = transforms.Compose(transform)


class BadCIFAR10(CIFAR10):
    def __init__(
        self, root, pidx, tlabel, trigger_loc, trigger_ptn, train=True, transform=None
    ):
        super(BadCIFAR10, self).__init__(root, train=train, transform=transform)
        self.pidx = pidx
        self.tlabel = tlabel
        self.bd_transform = transforms.Compose([AddTrigger(trigger_loc, trigger_ptn)])

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        poisoned = 0
        origin = target  # original target
        # Do the backdoor transformation first and then normal transformations.
        if self.pidx[index] == 1:
            img = self.bd_transform(img)
            target = self.tlabel
            poisoned = 1
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target, poisoned, origin

