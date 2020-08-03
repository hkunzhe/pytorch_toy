import argparse
from copy import deepcopy
import shutil
import os
import random
import time

import pandas as pd
import numpy as np
import setGPU
import torch
import torch.nn as nn
import yaml
from tabulate import tabulate
from torch.utils.data import DataLoader

from dataset import cifar
import resnet

random.seed(100)
np.random.seed(100)
torch.manual_seed(100)
torch.cuda.manual_seed(100)
torch.cuda.manual_seed_all(100)  # multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(model, train_loader, criterion, optimizer):
    model.train()
    tloss = 0  # total loss
    correct = 0

    for batch_idx, (data, target, poisoned, _) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda(
        optimizer.zero_grad()
        output = model(data)
        criterion.reduction = "none"
        raw_loss = criterion(output, target)
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


def test(test_model, test_loader, criterion):
    model = deepcopy(test_model)
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (data, target, _, _) in enumerate(test_loader):
        with torch.no_grad():
            data, target = data.cuda(), target.cuda()
            output = model(data)
            criterion.reduction = "mean"
            loss = criterion(output, target)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/example.yaml")
    args = parser.parse_args()

    assert os.path.exists(args.config)
    print("Load configuration file from {}".format(args.config))
    with open(args.config, "r") as f:
        print(f.read())
        f.seek(0)
        config = yaml.safe_load(f)

    assert os.path.exists(config["saved_path"])
    config_name = args.config.split("/")[-1].split(".")[0]
    saved_path = os.path.join(config["saved_path"], config_name)
    if not os.path.exists(saved_path):
        os.mkdir(saved_path)
    shutil.copy2(args.config, saved_path)

    # Training dataset and dataloader
    root = config["dataset_path"]
    pratio = config["poison_ratio"]
    tlabel = config["target_label"]
    trigger_loc = torch.tensor(config["trigger"])
    trigger_ptn = torch.randint(0, 256, [len(trigger_loc)])
    train_data = cifar.BadCIFAR10(root, pratio, tlabel, trigger_loc, trigger_ptn)
    train_data.set_transform(**config["train_transform"])
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)

    # Normal test dataset and dataloader
    test_data = cifar.CIFAR10(root, train=False)
    test_data.set_transform(**config["test_transform"])
    test_loader = DataLoader(test_data, batch_size=config["batch_size"], shuffle=False)

    # Backdoor test dataset and dataloader
    bd_test_data = cifar.BadCIFAR10(
        root, 1, tlabel, trigger_loc, trigger_ptn, train=False
    )
    bd_test_data.set_transform(**config["bd_test_transform"])
    bd_test_loader = DataLoader(
        bd_test_data, batch_size=config["batch_size"], shuffle=False
    )

    print("Start training...")
    model = resnet.ResNet18()
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    log = []
    for epoch in range(config["num_epoches"] + 1):
        start_time = time.time()
        train_results = train(model, train_loader, criterion, optimizer)
        bd_results = test(model, bd_test_loader, criterion)
        test_results = test(model, test_loader, criterion)
        elapse_time = time.time() - start_time

        row_dict = {
            "epoch": ["{}/{}".format(epoch + 1, config["num_epoches"])],
            "train_loss": [train_results["tloss"]],
            "train_acc(%)": [train_results["acc"]],
            "test_loss": [test_results["loss"]],
            "test_acc(%)": [test_results["acc"]],
            "bd_loss": [bd_results["loss"]],
            "bd_acc": [bd_results["acc"]],
            "time(s)": [elapse_time],
        }
        table = tabulate(row_dict, headers="keys", tablefmt="github", floatfmt="9.5f")
        if (epoch + 1) % 30 == 1:
            table = table.split("\n")
            table = "\n".join([table[1]] + table)
        else:
            table = table.split("\n")[2]
        print(table)

        log.append({k: v[0] for k, v in row_dict.items()})
        pd.DataFrame(log).to_csv(os.path.join(saved_path, "log.csv"), index=False)


if __name__ == "__main__":
    main()

