from collections import OrderedDict
from contextlib import suppress
from timm.utils import *
from torch import nn
import torch
import time
import timm
import os

tested_models = ['vit_tiny_patch16_224']


def get_test_loader():
    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
    dataset_eval = timm.data.create_dataset(
        '', root='/media/hlemarchant/Data/ImageNet', split='validation', is_training=False, batch_size=128)
    loader_eval = timm.data.create_loader(
        dataset_eval,
        input_size=[3, 224, 224],
        batch_size=128,
        is_training=False,
        use_prefetcher=True,
        interpolation='bicubic',
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        num_workers=4,
        distributed=distributed,
        crop_pct=0.9,
        pin_memory=False,
    )
    return loader_eval


def validate(model, loader, loss_fn, amp_autocast=suppress, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = 0
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            distributed = False
            if 'WORLD_SIZE' in os.environ:
                distributed = int(os.environ['WORLD_SIZE']) > 1

            if distributed:
                device = 'cuda:%d' % 0
                torch.cuda.set_device(0)
                torch.distributed.init_process_group(backend='nccl', init_method='env://')
                world_size = torch.distributed.get_world_size()

            if distributed:
                reduced_loss = reduce_tensor(loss.data, world_size)
                acc1 = reduce_tensor(acc1, world_size)
                acc5 = reduce_tensor(acc5, world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if last_batch or batch_idx % 100 == 0:
                log_name = 'Test' + log_suffix
                print(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics


def validate_attack(model, loader, loss_fn, amp_autocast=suppress, log_suffix=''):
    epsilonMax, clipMin, clipMax, targeted = 0.031, -1.0, 1.0, False
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx

            with amp_autocast():
                xDataTemp = input
                xDataTemp.requires_grad = True
                output = model(xDataTemp)
                model.zero_grad()
                cost = loss_fn(output, target)
                cost.backward()
                signDataGrad = xDataTemp.grad.data.sign()
                if targeted:
                    perturbedImage = input - epsilonMax * signDataGrad
                else:
                    perturbedImage = input + epsilonMax * signDataGrad
                perturbedImage = torch.clamp(perturbedImage, clipMin, clipMax)
                output = model(perturbedImage)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = 0
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            distributed = False
            if 'WORLD_SIZE' in os.environ:
                distributed = int(os.environ['WORLD_SIZE']) > 1

            if distributed:
                device = 'cuda:%d' % 0
                torch.cuda.set_device(0)
                torch.distributed.init_process_group(backend='nccl', init_method='env://')
                world_size = torch.distributed.get_world_size()

            if distributed:
                reduced_loss = reduce_tensor(loss.data, world_size)
                acc1 = reduce_tensor(acc1, world_size)
                acc5 = reduce_tensor(acc5, world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if last_batch or batch_idx % 100 == 0:
                log_name = 'Test' + log_suffix
                print(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics


if __name__ == '__main__':
    loader = get_test_loader()
    model = timm.create_model(tested_models[0],
                              checkpoint_path='output/train/20211005-100902-vit_tiny_patch16_224-224/checkpoint-100.pth.tar')
    model = model.cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()
    # metrics = validate(model, loader, validate_loss_fn)
    metrics = validate_attack(model, loader, validate_loss_fn)
    print(metrics)
