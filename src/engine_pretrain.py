# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200

    accum_iter = args['accum_iter']

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # set model epoch
    model.epoch = epoch
    for data_iter_step, (samples, _labels, _vids) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)



        #print(samples.shape)# 64x3x224x224 for img, 64x1x512x128 for audio
        samples = samples.to(device, non_blocking=True)
        
        # comment out when not debugging
        # from fvcore.nn import FlopCountAnalysis, parameter_count_table
        # if data_iter_step == 1:
        #     flops = FlopCountAnalysis(model, samples)
        #     print(flops.total())
        #     print(parameter_count_table(model))


        with torch.cuda.amp.autocast():
            loss_a, _, _, _ = model(samples, mask_ratio=args['mask_ratio'])
        loss_value = loss_a.item()
        loss_total = loss_a

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        #loss /= accum_iter
        loss_total = loss_total / accum_iter
        loss_scaler(loss_total, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= world_size
    return rt

def validate(model, val_loader, criterion, device, args):
    # if args.distributed:
    if args['distributed']:
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1
        
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, (x, _, _) in enumerate(val_loader):
            x = x.to(device)
            y = model(x)
            
            if isinstance(y, tuple):
                y = y[0]  
            
            loss = criterion(y, x)

            # if args.distributed:
            if args['distributed']:
                reduced_loss = reduce_tensor(loss.data, world_size)
                total_loss += reduced_loss.item()
            else:
                total_loss += loss.item()

    return total_loss / (i + 1)


##Validation
def validate_one_epoch(model: torch.nn.Module,
                       data_loader: Iterable,
                       device: torch.device, epoch: int,
                       log_writer=None,
                       args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Validation: [{}]'.format(epoch)
    print_freq = 200

    with torch.no_grad():
        for data_iter_step, (samples, _labels, _vids) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            samples = samples.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                loss_a, _, _, _ = model(samples, mask_ratio=args['mask_ratio'])
            loss_value = loss_a.item()

            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)

            loss_value_reduce = misc.all_reduce_mean(loss_value)

            if log_writer is not None:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar('val_loss', loss_value_reduce, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged validation stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
