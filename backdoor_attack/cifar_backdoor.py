import argparse
import copy
import random
import os

os.environ["MODIN_ENGINE"] = "ray"

import matplotlib.pyplot as plt

import modin.pandas as pd
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


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target, poisoned = t(img, target)
        return img, target, poisoned


class AddBackdoor(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, img, target):
        return add_backdoor(img, target, **self.kwargs)


def add_backdoor(img, target, **kwargs):
    trigger_loc = kwargs["trigger_loc"]
    trigger_ptn = kwargs["trigger_ptn"]
    t = kwargs["target_label"]
    s = kwargs["source_label"]
    p = kwargs["poison_ratio"]
    poisoned = 0
    if random.random() < p and target == s:
        poisoned = 1
        for i, (m, n) in enumerate(trigger_loc):
            img[m, n, :] = trigger_ptn[i]  # add trigger
            target = t  # label-flipping
    return img, target, poisoned


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


def train(model, train_loader, epoch, **kwargs):
    device = torch.device("cuda")
    model = model.to(device)
    model.train()
    batch_size = train_loader.batch_size
    optim = torch.optim.Adam(model.parameters(), lr=kwargs["lr"])

    loss_list = np.zeros(len(train_loader.dataset))
    target_list = np.zeros(len(train_loader.dataset))
    pred_list = np.zeros(len(train_loader.dataset))
    poisoned_list = np.zeros(len(train_loader.dataset))
    for batch_idx, (data, target, poisoned) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        poisoned = poisoned.float()
        optim.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target, reduction="none")
        pred = output.argmax(dim=1)

        if len(data) == batch_size:
            loss_list[batch_idx * batch_size : (batch_idx + 1) * batch_size] = (
                loss.detach().cpu().numpy()
            )
            target_list[batch_idx * batch_size : (batch_idx + 1) * batch_size] = (
                target.detach().cpu().numpy()
            )
            pred_list[batch_idx * batch_size : (batch_idx + 1) * batch_size] = (
                pred.detach().cpu().numpy()
            )
            poisoned_list[batch_idx * batch_size : (batch_idx + 1) * batch_size] = (
                poisoned.detach().cpu().numpy()
            )
        else:
            loss_list[batch_idx * batch_size : len(train_loader.dataset)] = (
                loss.detach().cpu().numpy()
            )
            target_list[batch_idx * batch_size : len(train_loader.dataset)] = (
                target.detach().cpu().numpy()
            )
            pred_list[batch_idx * batch_size : len(train_loader.dataset)] = (
                pred.detach().cpu().numpy()
            )
            poisoned_list[
                batch_idx * batch_size : len(train_loader.dataset)
            ] = poisoned.cpu().numpy()

        reduced_loss = torch.mean(loss)
        reduced_loss.backward()
        optim.step()

    logs = pd.DataFrame(
        {
            "label": target_list,
            "prediction": pred_list,
            "poisoned": poisoned_list,
            "loss": loss_list,
        }
    )
    logs_path = os.path.join(kwargs["saved_path"], "logs")
    if not os.path.exists(logs_path):
        os.mkdir(logs_path)
    logs.to_parquet(os.path.join(logs_path, "epoch_{}.parq".format(epoch)))


def test(model, test_loader, epoch):
    device = torch.device("cuda")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target, _) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)

            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= batch_idx
    acc = 100.0 * correct / len(test_loader.dataset)

    return test_loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/cifar.yaml")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError("No such file: {}".format(args.config))

    print("Load configuration file from {}".format(args.config))
    with open(args.config, "r") as f:
        config = yaml.safe_load(f.read())

    if not os.path.exists(config["saved_path"]):
        os.mkdir(config["saved_path"])
    saved_config_path = os.path.join(config["saved_path"], args.config.split("/")[-1])
    if not os.path.exists(saved_config_path):
        os.mkdir(saved_config_path)

    train_config = {"lr": config["lr"], "saved_path": saved_config_path}
    bd_train_config = {
        "trigger_loc": torch.tensor(config["trigger"]["client"]),
        "trigger_ptn": torch.randint(0, 255, [len(config["trigger"]["client"])]),
        "target_label": config["target_label"],
        "source_label": config["source_label"],
        "poison_ratio": config["poison_ratio"],
    }
    bd_test_config = copy.deepcopy(bd_train_config)
    bd_test_config["poison_ratio"] = 1

    # Transform
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
    bd_train_transform = Compose([AddBackdoor(**bd_train_config)])
    bd_test_transform = Compose([AddBackdoor(**bd_test_config)])

    # DataLoader
    train_data = cifar.CIFAR10(
        data_dir="./cifar10/cifar-10-batches-py",
        idx=0,
        indices=[50000],
        transform=train_transform,
        backdoor_transform=bd_train_transform,
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
        backdoor_transform=bd_test_transform,
    )
    train_loader = DataLoader(train_data, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=512, shuffle=True)
    bd_test_loader = DataLoader(bd_test_data, batch_size=512, shuffle=True)

    model = resnet.ResNet18()

    for epoch in trange(300, desc="training", unit="epoch"):
        train(model, train_loader, epoch, **train_config)
        test_loss, test_acc = test(model, test_loader, epoch)
        bd_loss, bd_acc = test(model, bd_test_loader, epoch)
        print("test loss: {}, test accuracy: {}".format(test_loss, test_acc))
        print("backdoor loss: {}, backdoor accuracy: {}".format(bd_loss, bd_acc))


if __name__ == "__main__":
    main()

