import argparse
import copy
import os
import random
import time

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
from tabulate import tabulate
from torch.utils.data import DataLoader

from dataset import cifar
import resnet
from utils import AddTrigger

random.seed(100)
np.random.seed(100)
torch.manual_seed(100)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(model, train_loader, device, optimizer):
    model.train()
    tloss = 0  # total loss
    correct = 0

    for batch_idx, (data, target, poisoned) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        raw_loss = F.cross_entropy(output, target, reduction="none")
        loss = torch.mean(raw_loss)
        loss.backward()
        optimizer.step()

        # stats
        tloss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    stats = {
        "tloss": tloss / batch_idx,
        "acc": 100.0 * correct / len(train_loader.dataset),
    }
    return stats


def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target, _) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)

            # stats
            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    stats = {
        "loss": test_loss / batch_idx,
        "acc": 100.0 * correct / len(test_loader.dataset),
    }
    return stats


def main():
    # Config
    print("Configuring...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/example.yaml")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError("No such file: {}".format(args.config))

    print("Load configuration file from {}".format(args.config))
    with open(args.config, "r") as f:
        config = yaml.safe_load(f.read())

    trigger_config = {
        "trigger_loc": torch.tensor(config["trigger"]),
        "trigger_ptn": torch.randint(0, 256, [len(config["trigger"])]),
    }
    bd_train_config = {
        "tlabel": config["target_label"],
        "pratio": config["poison_ratio"],
        "bd_transform": transforms.Compose([AddTrigger(**trigger_config)]),
    }
    bd_test_config = copy.deepcopy(bd_train_config)
    bd_test_config["pratio"] = 1

    # Transform
    print("Prepare data...")
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

    # DataLoader
    train_data = cifar.CIFAR10(
        root=config["dataset_path"], transform=train_transform, **bd_train_config,
    )
    test_data = cifar.CIFAR10(
        root=config["dataset_path"], train=False, transform=test_transform,
    )
    bd_test_data = cifar.CIFAR10(
        root=config["dataset_path"],
        train=False,
        transform=test_transform,
        **bd_test_config,
    )
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config["batch_size"], shuffle=True)
    bd_test_loader = DataLoader(
        bd_test_data, batch_size=config["batch_size"], shuffle=True
    )

    print("Start training...")
    device = torch.device("cuda")
    model = resnet.ResNet18()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(config["num_epoches"] + 1):
        start_time = time.time()
        train_stats = train(model, train_loader, device, optimizer)
        test_stats = test(model, test_loader, device)
        bd_stats = test(model, bd_test_loader, device)
        elapse_time = time.time() - start_time

        stats = {
            "epoch": ["{}/{}".format(epoch + 1, config["num_epoches"])],
            "train_loss": [train_stats["tloss"]],
            "train_acc(%)": [train_stats["acc"]],
            "test_loss": [test_stats["loss"]],
            "test_acc(%)": [test_stats["acc"]],
            "bd_loss": [bd_stats["loss"]],
            "bd_acc": [bd_stats["acc"]],
            "time(s)": [elapse_time],
        }
        table = tabulate(stats, headers="keys", tablefmt="github", floatfmt="9.5f")
        if (epoch + 1) % 30 == 1:
            table = table.split("\n")
            table = "\n".join([table[1]] + table)
        else:
            table = table.split("\n")[2]
        print(table)


if __name__ == "__main__":
    main()
