import argparse
import shutil
import os
import time

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from data import cifar
from model import resnet_cifar
from setup import load_config, get_saved_dir, get_storage_dir
from torchattacks.attacks import PGD
from utils import AverageMeter, tabulate_step_meter, tabulate_epoch_meter, resume_state


def train(model, loader, criterion, optimizer, gpu, attacker=None):
    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("acc")

    model.train()
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(loader):
        if attacker is not None:
            data = attacker(data, target).cuda(gpu, non_blocking=True)
        else:
            data = data.cuda(gpu, non_blocking=True)
        target = target.cuda(gpu, non_blocking=True)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update meters.
        loss_meter.update(loss.item())
        pred = output.argmax(dim=1, keepdim=True)
        truth = pred.view_as(target).eq(target)
        acc_meter.update((1.0 * torch.sum(truth) / len(truth)).item())

        tabulate_step_meter(batch_idx, len(loader), 100, *[loss_meter, acc_meter])

    print("Summary...")
    elapsed_time = time.time() - start_time
    tabulate_epoch_meter(elapsed_time, *[loss_meter, acc_meter])

    result = {"loss": loss_meter.total_avg, "acc": acc_meter.total_avg}
    return result


def test(model, loader, criterion, gpu, attacker=None):
    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("acc")

    model.eval()
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(loader):
        if attacker is not None:
            print("Adv test...")
            data = attacker(data, target).cuda(gpu, non_blocking=True)
        else:
            print("Clean test...")
            data = data.cuda(gpu, non_blocking=True)
        target = target.cuda(gpu, non_blocking=True)

        with torch.no_grad():
            output = model(data)
            loss = criterion(output, target)

        # Update meters.
        loss_meter.update(loss.item())
        pred = output.argmax(dim=1, keepdim=True)
        truth = pred.view_as(target).eq(target)
        acc_meter.update((1.0 * torch.sum(truth) / len(truth)).item())

        tabulate_step_meter(batch_idx, len(loader), 25, *[loss_meter, acc_meter])

    print("Summary...")
    elapsed_time = time.time() - start_time
    tabulate_epoch_meter(elapsed_time, *[loss_meter, acc_meter])

    result = {"loss": loss_meter.total_avg, "acc": acc_meter.total_avg}
    return result


def main():
    # Setup.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/example.yaml")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="path to checkpoint or False (means training from scratch)",
    )
    args = parser.parse_args()
    config, inner_dir, config_name = load_config(args.config)
    saved_dir = get_saved_dir(config, inner_dir, config_name, args.resume)
    storage_dir, ckpt_dir = get_storage_dir(config, inner_dir, config_name, args.resume)

    # Prepare data.
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
    train_data = cifar.CIFAR10(root=config["dataset_dir"], transform=train_transform)
    test_data = cifar.CIFAR10(
        root=config["dataset_dir"], train=False, transform=test_transform
    )
    train_loader = DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True, num_workers=4
    )
    test_loader = DataLoader(test_data, batch_size=config["batch_size"], num_workers=4)

    model = resnet_cifar.ResNet18()
    gpu = int(args.gpu)
    print("Set GPU to {}".format(args.gpu))
    model = model.cuda(gpu)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), **config["optimizer"]["SGD"])
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, **config["lr_scheduler"]["MultiStepLR"]
    )
    resumed_epoch = resume_state(model, optimizer, args.resume, ckpt_dir, scheduler)
    attacker = PGD(model, eps=8 / 255, alpha=2 / 255, steps=7, random_start=True)

    for epoch in range(config["num_epochs"] - resumed_epoch):
        print("Epoch: {}/{}".format(epoch + resumed_epoch + 1, config["num_epochs"]))
        print("Adversarial training...")
        adv_train_result = train(
            model, train_loader, criterion, optimizer, gpu, attacker=attacker
        )
        if scheduler is not None:
            scheduler.step()
            print("Adjust learning rate to {}".format(optimizer.param_groups[0]["lr"]))
        print("Test model on clean data...")
        clean_test_result = test(model, test_loader, criterion, gpu)
        print("Test model on adversarial data...")
        adv_test_result = test(model, test_loader, criterion, gpu, attacker=attacker)
        result = {
            "adv_train": adv_train_result,
            "clean_test": clean_test_result,
            "adv_test": adv_test_result,
        }

        # Save checkpoint
        saved_dict = {
            "epoch": epoch + resumed_epoch + 1,
            "result": result,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if scheduler is not None:
            saved_dict["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(
            saved_dict,
            os.path.join(ckpt_dir, "epoch{}.pt".format(epoch + resumed_epoch + 1)),
        )


if __name__ == "__main__":
    main()
