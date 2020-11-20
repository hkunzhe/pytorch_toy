import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from data import cifar
from model import resnet_cifar
from model.utils import NormalizeByChannelMeanStd, resume_state
from utils.setup import load_config, get_saved_dir, get_storage_dir, get_logger
from utils.torchattacks.attacks import PGD
from utils.trainer import train, test

torch.backends.cudnn.benchmark = True


def main():
    # Setup.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/example.yaml")
    parser.add_argument("--gpu", default="0", type=str)
    # Path to checkpoint (empty string means the latest checkpoint)
    # or False (means training from scratch).
    parser.add_argument("--resume", default="", type=str)
    args = parser.parse_args()
    config, inner_dir, config_name = load_config(args.config)
    saved_dir = get_saved_dir(config, inner_dir, config_name, args.resume)
    storage_dir, ckpt_dir = get_storage_dir(config, inner_dir, config_name, args.resume)
    logger = get_logger(saved_dir, "adv_training.log", args.resume)

    # Prepare data.
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_data = cifar.CIFAR10(root=config["dataset_dir"], transform=train_transform)
    test_data = cifar.CIFAR10(
        root=config["dataset_dir"], train=False, transform=test_transform
    )
    train_loader = DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True, num_workers=4
    )
    test_loader = DataLoader(test_data, batch_size=config["batch_size"], num_workers=4)

    # Resume training state.
    model = resnet_cifar.ResNet18()
    gpu = int(args.gpu)
    logger.info("Set GPU to {}".format(args.gpu))
    model = model.cuda(gpu)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), **config["optimizer"]["SGD"])
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, **config["lr_scheduler"]["MultiStepLR"]
    )
    resumed_epoch = resume_state(model, optimizer, args.resume, ckpt_dir, scheduler)

    # Set attack first and then add a normalized layer.
    pgd_config = {}
    for k, v in config["pgd_attack"].items():
        if k == "eps" or k == "alpha":
            pgd_config[k] = eval(v)
        else:
            pgd_config[k] = v
    attacker = PGD(model, **pgd_config)
    normalize_net = NormalizeByChannelMeanStd(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    )
    normalize_net.cuda(gpu)
    model = nn.Sequential(normalize_net, model)

    for epoch in range(config["num_epochs"] - resumed_epoch):
        logger.info(
            "===Epoch: {}/{}===".format(epoch + resumed_epoch + 1, config["num_epochs"])
        )
        logger.info("Adversarial training...")
        adv_train_result = train(
            model, train_loader, criterion, optimizer, logger, attacker=attacker
        )
        if scheduler is not None:
            scheduler.step()
            logger.info(
                "Adjust learning rate to {}".format(optimizer.param_groups[0]["lr"])
            )
        logger.info("Test model on clean data...")
        clean_test_result = test(model, test_loader, criterion, logger)
        logger.info("Test model on adversarial data...")
        adv_test_result = test(model, test_loader, criterion, logger, attacker=attacker)
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
