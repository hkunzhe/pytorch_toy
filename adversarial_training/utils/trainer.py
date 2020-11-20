import time

import torch

from .log import AverageMeter, tabulate_step_meter, tabulate_epoch_meter


def train(model, loader, criterion, optimizer, logger, attacker=None):
    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("acc")

    model.train()
    gpu = next(model.parameters()).device
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

        tabulate_step_meter(
            batch_idx, len(loader), 100, [loss_meter, acc_meter], logger
        )

    logger.info("Summary...")
    elapsed_time = time.time() - start_time
    tabulate_epoch_meter(elapsed_time, [loss_meter, acc_meter], logger)

    result = {"loss": loss_meter.total_avg, "acc": acc_meter.total_avg}
    return result


def test(model, loader, criterion, logger, attacker=None):
    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("acc")

    model.eval()
    gpu = next(model.parameters()).device
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(loader):
        if attacker is not None:
            data = attacker(data, target).cuda(gpu, non_blocking=True)
        else:
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

        tabulate_step_meter(batch_idx, len(loader), 25, [loss_meter, acc_meter], logger)

    logger.info("Summary...")
    elapsed_time = time.time() - start_time
    tabulate_epoch_meter(elapsed_time, [loss_meter, acc_meter], logger)

    result = {"loss": loss_meter.total_avg, "acc": acc_meter.total_avg}
    return result
