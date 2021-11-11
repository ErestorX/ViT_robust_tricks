from matplotlib.ticker import PercentFormatter
from torchvision.utils import save_image
from collections import OrderedDict
import matplotlib.pyplot as plt
from contextlib import suppress
from timm.utils import *
from torch import nn
import numpy as np
import argparse
import models
import torch
import time
import timm
import cv2
import csv
import os


parser = argparse.ArgumentParser()
parser.add_argument('--version', default=0, type=int)
parser.add_argument('--ckpt', default='', type=str)
parser.add_argument('-p', action='store_true', default=False)


def get_val_loader():
    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
    dataset_eval = timm.data.create_dataset(
        '', root='/media/hlemarchant/Data/ImageNet', split='validation', is_training=False, batch_size=128)
    loader_eval = timm.data.create_loader(
        dataset_eval,
        input_size=[3, 224, 224],
        batch_size=64,
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


def validate(model, loader, loss_fn, val_path):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    cleanImage = None
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            cleanImage = input
            last_batch = batch_idx == last_idx

            with suppress():
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
                log_name = 'Clean'
                print('{0}: [{1:>4d}/{2}]  Acc@1: {top1.avg:>7.4f}'.format(log_name, batch_idx, last_idx,
                                                                           batch_time=batch_time_m, top1=top1_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])
    save_image((cleanImage[:16] + 1) / 2, val_path + '/clean_batch.png')

    return metrics


def validate_attack(model, loader, loss_fn, val_path):
    epsilonMax = 0.03
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    perturbedImage = None
    perturbationVal = None
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx

        with suppress():
            input.requires_grad = True
            output = model(input)
            cost = loss_fn(output, target)
            grad = torch.autograd.grad(cost, input, retain_graph=False, create_graph=False)[0]
            perturbationVal = grad.sign()
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
            log_name = 'Adversarial'
            print('{0}: [{1:>4d}/{2}]  Acc@1: {top1.avg:>7.4f}'.format(log_name, batch_idx, last_idx,
                                                                       batch_time=batch_time_m, top1=top1_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])
    save_image((perturbedImage[:16] + 1) / 2, val_path + '/adv_batch.png')
    save_image(perturbationVal[:16], val_path + '/perturb_batch.png')

    return metrics


def average_q_px_dist_per_head_per_block(title, fname, loader, model):
    def get_features(name):
        def hook(model, input, output):
            qkvs[name] = output.detach()
        return hook

    for block_id, block in enumerate(model.blocks):
        block.attn.qkv.register_forward_hook(get_features(str(block_id)))

    patch_size = 16 if '16' in title.split('_')[2] else 32
    model.eval()
    qkvs = {}
    for batch_idx, (input, target) in enumerate(loader):
        _ = model(input)
        for block, qkv in qkvs.items():
            num_heads, scale = model.blocks[0].attn.num_heads, model.blocks[0].attn.scale
            B, N, CCC = qkv.shape
            C = CCC//3
            qkv = qkv.reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            _, H, _, _ = attn.shape
            attn = attn.permute(1, 0, 2, 3)
            vect = torch.arange(N).reshape((1, N))
            dist_map = torch.sqrt(((vect - torch.transpose(vect, 0, 1)) % (N-1)**0.5) ** 2 + ((vect - torch.transpose(vect, 0, 1)) // (N-1)**0.5) ** 2)
            per_head_dist_map = torch.sum(attn * torch.as_tensor(dist_map).to(device='cuda'), (1, 2, 3))/torch.sum(attn, (1, 2, 3))
            qkvs[block] = per_head_dist_map * patch_size
        break
    vals = []
    for qkv in qkvs.values():
        vals.append(qkv.cpu().numpy())
    vals = np.asarray(vals)
    block_names = [str(i) for i in range(len(vals))]
    fig, ax = plt.subplots()
    for head in range(len(vals[0])):
        ax.scatter(block_names, vals[:, head], label='head_'+str(head))
    fig.suptitle(title)
    if len(vals[0]) < 7:
        ax.legend()
    ax.set_ylabel('Attention distance in Pixel')
    ax.set_xlabel('Block id')
    ax.grid(True, which='both')
    ax.set_ylim(ymax=180, ymin=0)
    plt.savefig(fname + '/Attn_dist.png')
    plt.close()


def freq_hist(title, val_path):
    for img in ['clean', 'adv', 'perturb']:
        image = cv2.imread(val_path + '/' + img + '_batch.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        data = image.ravel()

        _ = plt.hist(data, weights=np.ones(len(data)) / len(data), bins=256, color='orange', )
        _ = plt.title(title)
        _ = plt.xlabel('Intensity Value')
        _ = plt.ylabel('Count')
        _ = plt.ylim(top=.2 if img == 'perturb' else .01)
        _ = plt.legend(['Total'])
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.savefig(val_path + '/Pixel_hist_' + img + '.png')
        plt.close()
        plt.imsave(val_path + '/Freq_' + img + '.png', np.log(abs(np.fft.fftshift(np.fft.fft2(image)))), cmap='gray')


def main():
    args = parser.parse_args()
    custom_model = args.ckpt != ''
    train_path = 'output/train/'
    val_path = 'output/val/'
    ext = '/model_best.pth.tar'
    tested_models = ['vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_small_patch32_224', 'vit_base_patch16_224',
                     'vit_base_patch32_224']
    if args.version < 0 or args.version >= len(tested_models):
        print("Error: Version asked does not exist.")
        return
    if args.ckpt == '':
        ckpt_path = train_path + tested_models[args.version]
        val_path = val_path + tested_models[args.version] + ('_pretrained' if args.p else '_scratch')
    else:
        ckpt_path = train_path + tested_models[args.version] + '_' + args.ckpt
        val_path = val_path + tested_models[args.version] + '_' + args.ckpt
    if os.path.exists(ckpt_path) or args.p:
        if not os.path.exists(val_path):
            os.mkdir(val_path)
            ckpt_file = ckpt_path + ext
            if args.p and not custom_model:
                model = timm.create_model(tested_models[args.version], pretrained=True)
            else:
                model = timm.create_model('custom_' + tested_models[args.version] if custom_model else tested_models[args.version], checkpoint_path=ckpt_file)
            loader = get_val_loader()
            model = model.cuda()
            validate_loss_fn = nn.CrossEntropyLoss().cuda()

            average_q_px_dist_per_head_per_block(val_path.split('/')[-1], val_path, loader, model)
            clean_metrics = validate(model, loader, validate_loss_fn, val_path)
            adv_metrics = validate_attack(model, loader, validate_loss_fn, val_path)
            # freq_hist(val_path.split('/')[-1], val_path)

            with open(val_path + '/Validation.csv', 'w+', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['Data', 'Loss', 'Acc@1', 'Acc@5'])
                writer.writerow(['Clean', clean_metrics['loss'], clean_metrics['top1'], clean_metrics['top5']])
                writer.writerow(['Adv', adv_metrics['loss'], adv_metrics['top1'], adv_metrics['top5']])
        else:
            print("Error: Results already existing:", val_path.split('/')[-1])
    else:
        print("Error: Model asked does not exist:", ckpt_path.split('/')[-1])
        if not custom_model:
            print("\tOnly the pretrained version is available.")


if __name__ == '__main__':
    main()
