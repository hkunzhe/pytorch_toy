import random

import matplotlib.pyplot as plt
import numpy as np
import setGPU
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

import cifar
import resnet

random.seed(100)
np.random.seed(100)
torch.manual_seed(100)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Since the built-in torchvision.transfroms.Compose only accept one args. 
# Customized Compose accepts two args for img and target.
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target

# Customize adding backdoor to transforms that speeds up preprocess.
class AddBackdoor(object):
    def __init__(self, trigger_loc, trigger_ptn, target_label=7, p=0.05):
        self.trigger_loc = trigger_loc
        self.trigger_ptn = trigger_ptn
        self.target_label = target_label
        self.p = p

    def __call__(self, img, target):
        return add_backdoor(
            img, target, self.trigger_loc, self.trigger_ptn, self.target_label, self.p
        )


def add_backdoor(img, target, trigger_loc, trigger_ptn, target_label=7, p=0.05):
    if random.random() < p:
        for i, (m, n) in enumerate(trigger_loc):
            img[m, n, :] = trigger_ptn[i]  # add trigger
            target = target_label  # label-flipping
    return img, target


def plot_data(dataset, nrows=10, ncols=10):
    data_loader = DataLoader(dataset, batch_size=nrows * ncols)
    batch_data, targets = next(iter(data_loader))  # fetch the first batch data
    batch_data = batch_data.permute(0, 2, 3, 1)  # convert to NHWC
    fig, axes = plt.subplots(nrows, ncols)
    fig.figsize = (12, 9)
    fig.dpi = 600
    for r in range(nrows):
        for c in range(ncols):
            idx = r * nrows + c
            axes[r][c].imshow(batch_data[idx].numpy())
            axes[r][c].set_title(str(targets[idx].item()))
            axes[r][c].axis("off")
    fig.savefig("test.png")


def train(model, train_loader, epoch):
    device = torch.device("cuda")
    model = model.to(device)
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optim.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optim.step()

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss /= len(train_loader.dataset)
    acc = 100.0 * correct / len(train_loader.dataset)

    return train_loss, acc


def test(model, test_loader, epoch):
    device = torch.device("cuda")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)

            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100.0 * correct / len(test_loader.dataset)

    return test_loss, acc


def main():
    with open("./config/cifar10.yaml", "r") as f:
        config = yaml.safe_load(f.read())

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    trigger_loc = torch.tensor(config["trigger"]["client"])
    trigger_ptn = torch.randint(0, 255, [len(trigger_loc)])
    bd_transform = Compose([AddBackdoor(trigger_loc, trigger_ptn, p=0.05)])
    bd_test_transform = Compose([AddBackdoor(trigger_loc, trigger_ptn, p=1)])
    train_data = cifar.CIFAR10(
        data_dir="./cifar10/cifar-10-batches-py",
        idx=0,
        indices=[50000],
        transform=train_transform,
        backdoor_transform=bd_transform,
    )
    test_data = cifar.CIFAR10(
        data_dir="./cifar10/cifar-10-batches-py",
        evaluation=True,
        transform=test_transform,
    )
    bd_test_data = cifar.CIFAR10(
        data_dir="./cifar10/cifar-10-batches-py",
        evaluation=True,
        transform=test_transform,
        backdoor_transform=bd_transform,
    )
    train_loader = DataLoader(train_data, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=512, shuffle=True)
    bd_test_loader = DataLoader(bd_test_data, batch_size=512, shuffle=True)

    model = resnet.ResNet18()

    for epoch in trange(100, desc="training", unit="epoch"):
        train_loss, train_acc = train(model, train_loader, epoch)
        test_loss, test_acc = test(model, test_loader, epoch)
        bd_loss, bd_acc = test(model, bd_test_loader, epoch)
        print("train loss: {}, train accuracy: {}".format(train_loss, train_acc))
        print("test loss: {}, test accuracy: {}".format(test_loss, test_acc))
        print("backdoor loss: {}, backdoor accuracy: {}".format(bd_loss, bd_acc))


if __name__ == "__main__":
    main()
