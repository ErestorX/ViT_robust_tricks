from collections import OrderedDict
import matplotlib.pyplot as plt
from contextlib import suppress
from timm.utils import *
from torch import nn
import numpy as np
import torch
import time
import timm
import os

tested_models = ['vit_tiny_patch16_224']


def get_val_loader():
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
    epsilonMax = 0.03
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx

        with amp_autocast():
            input.requires_grad = True
            output = model(input)
            cost = loss_fn(output, target)
            grad = torch.autograd.grad(cost, input, retain_graph=False, create_graph=False)[0]
            perturbedImage = input + epsilonMax * grad.sign()
            perturbedImage = torch.clamp(perturbedImage, -1, 1)
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

def average_q_px_dist_per_head_per_block(loader, model):
    def get_features(name):
        def hook(model, input, output):
            qkvs[name] = output.detach()
        return hook

    for block_id, block in enumerate(model.blocks):
        block.attn.qkv.register_forward_hook(get_features(str(block_id)))

    model.eval()
    qkvs = {}
    for batch_idx, (input, target) in enumerate(loader):
        _ = model(input)
        for block, qkv in qkvs.items():
            num_heads, scale = model.blocks[0].attn.num_heads, model.blocks[0].attn.scale
            patch_size = 16
            B, N, CCC = qkv.shape
            C = CCC//3
            qkv = qkv.reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            _, H, _, _ = attn.shape
            attn = attn.permute(1, 0, 2, 3)
            vect = torch.arange(N).reshape((1, N))
            dist_map = torch.sqrt(((vect - torch.transpose(vect, 0, 1)) % N**0.5) ** 2 + ((vect - torch.transpose(vect, 0, 1)) // N**0.5) ** 2)
            per_head_dist_map = torch.mean(attn * torch.as_tensor(dist_map).to(device='cuda'), (1, 2, 3)) * patch_size
            qkvs[block] = per_head_dist_map
        break
    vals = []
    for qkv in qkvs.values():
        vals.append(qkv.cpu().numpy())
    vals = np.asarray(vals)
    block_names = [str(i) for i in range(len(vals))]
    fig, ax = plt.subplots()
    for head in range(len(vals[0])):
        ax.scatter(block_names, vals[:, head], label='head_'+str(head))
    fig.suptitle('Average attention distance per head per block')
    ax.legend()
    ax.set_ylabel('Attention distance in Pixel')
    ax.set_xlabel('Block id')
    fig.show()


if __name__ == '__main__':
    loader = get_val_loader()
    model = timm.create_model(tested_models[0], pretrained=True)
    model = model.cuda()
    average_q_px_dist_per_head_per_block(loader, model)
    # validate_loss_fn = nn.CrossEntropyLoss().cuda()
    # metrics = validate(model, loader, validate_loss_fn)
    # print("Clean top1", metrics['top1'])
    # metrics = validate_attack(model, loader, validate_loss_fn)
    # print("Adversarial top1", metrics['top1'])
