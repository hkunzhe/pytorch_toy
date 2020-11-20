import os
from functools import cmp_to_key

import torch
import torch.nn as nn


class NormalizeByChannelMeanStd(nn.Module):
    """Normalizing the input to the network.
    """

    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        mean = self.mean[None, :, None, None]
        std = self.std[None, :, None, None]
        return tensor.sub(mean).div(std)

    def extra_repr(self):
        return "mean={}, std={}".format(self.mean, self.std)


def ckpt_key():
    def ckpt_cmp(x, y):
        epoch_x = int(x.split("h")[1].split(".")[0])
        epoch_y = int(y.split("h")[1].split(".")[0])
        return epoch_x - epoch_y

    return cmp_to_key(ckpt_cmp)


def resume_state(model, optimizer, resume, ckpt_dir, scheduler=None):
    """ Resume training state from checkpoint.

    Args:
        model (torch.nn.Module): Model to resume.
        optimizer (torch.optim): Optimizer to resume.
        resume (str): Checkpoint name or False (means training from scratch).
        ckpt_dir (str): Checkpoint directory.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler to resume (default: None).
    
    Returns:
        resumed_epoch (int): The epoch to resume (0 means training from scratch.)
    """
    file_list = sorted(os.listdir(ckpt_dir), key=ckpt_key())
    if resume == "False":
        print("Don't resume training state.")
        resumed_epoch = 0
        return resumed_epoch
    elif resume == "" and file_list:
        resumed_model = os.path.join(ckpt_dir, file_list[-1])
        print(
            "Resume training state from the latest checkpoint: {}".format(resumed_model)
        )
    else:
        resumed_model = os.path.join(ckpt_dir, resume)
        print("Resume training state from: {}".format(resumed_model))
    ckpt = torch.load(resumed_model)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print("Result of the resumed model: {}".format(ckpt["result"]))

    return ckpt["epoch"]
