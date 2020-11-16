import os
from functools import cmp_to_key

import torch
from tabulate import tabulate


class AverageMeter(object):
    """Computes and stores the average and current value.

    Modified from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.batch_avg = 0
        self.total_avg = 0
        self.sum = 0
        self.count = 0

    def update(self, avg, n=1):
        self.batch_avg = avg
        self.sum += avg * n
        self.count += n
        self.total_avg = self.sum / self.count


def tabulate_step_meter(batch_idx, num_batches, step_interval, *meters):
    """ Tabulate current average value of meters every `step_interval`.

    Args:
        batch_idx (int): The batch index in an epoch.
        num_batches (int): The number of batch in an epoch.
        step_interval (int): The step interval to tabulate.
        *meters (list or tuple of AverageMeter): A list of meters.
    """
    if batch_idx % step_interval == 0:
        step_meter = {"Iteration": ["{}/{}".format(batch_idx, num_batches)]}
        for m in meters:
            step_meter[m.name] = [m.batch_avg]
        table = tabulate(step_meter, headers="keys", tablefmt="github", floatfmt=".4f")
        if batch_idx == 0:
            table = table.split("\n")
            table = "\n".join([table[1]] + table)
        else:
            table = table.split("\n")[2]
        print(table)


def tabulate_epoch_meter(elapsed_time, *meters):
    """ Tabulate total average value of meters every epoch.

    Args:
        eplased_time (float): The elapsed time of a epoch.
        *meters (list or tuple of AverageMeter): A list of meters.
    """
    epoch_meter = {}
    for m in meters:
        epoch_meter[m.name] = [m.total_avg]
    epoch_meter["time"] = [elapsed_time]
    table = tabulate(epoch_meter, headers="keys", tablefmt="github", floatfmt=".4f")
    table = table.split("\n")
    table = "\n".join([table[1]] + table)
    print(table)


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
        resumed_epoch: The epoch to resume (0 means training from scratch.)
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
