from utils_eval import average_q_px_dist_per_head_per_block, freq_hist, get_CKA, get_adversarial_CKA
from torchvision.utils import save_image
from collections import OrderedDict
from contextlib import suppress
from timm.utils import *
from torch import nn
import argparse
import models
import torch
import time
import timm
import csv
import os


parser = argparse.ArgumentParser()
parser.add_argument('--data', default='/home2/hugo/ImageNet', type=str)
parser.add_argument('--version', default=0, type=int)
parser.add_argument('--ckpt', default='', type=str)
parser.add_argument('-p', action='store_true', default=False)


def get_val_loader(data_path, batch_size=64):
    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
    dataset_eval = timm.data.create_dataset('', root=data_path, split='validation', is_training=False, batch_size=128)
    loader_eval = timm.data.create_loader(
        dataset_eval,
        input_size=[3, 224, 224],
        batch_size=batch_size,
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


def main():
    args = parser.parse_args()
    args.data = None
    custom_model = args.ckpt != ''
    train_path = 'output/train/'
    val_path = 'output/val/'
    ext = '/model_best.pth.tar'
    tested_models = ['vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_small_patch32_224', 'vit_base_patch16_224',
                     'vit_base_patch32_224']
    custom_versions = ['doexp5', 'dosq4015']
    if args.version < 0 or args.version >= len(tested_models):
        print("Error: Version asked does not exist.")
        return
    if args.ckpt == '':
        ckpt_path = train_path + tested_models[args.version]
        val_path = val_path + tested_models[args.version] + ('_pretrained' if args.p else '_scratch')
        exp_name = tested_models[args.version] + ('_pretrained' if args.p else '_scratch')
    else:
        ckpt_path = train_path + tested_models[args.version] + '_' + args.ckpt
        val_path = val_path + tested_models[args.version] + '_' + args.ckpt
        exp_name = 'custom_' + tested_models[args.version] + '_' + args.ckpt
    loader = get_val_loader(args.data, batch_size=128)
    if os.path.exists(ckpt_path) or args.p:
        if not os.path.exists(val_path):
            os.mkdir(val_path)
            ckpt_file = ckpt_path + ext
            if args.p and not custom_model:
                model = timm.create_model(tested_models[args.version], pretrained=True)
            else:
                model = timm.create_model('custom_' + tested_models[args.version] if custom_model else tested_models[args.version], checkpoint_path=ckpt_file)
            model = model.cuda()
            validate_loss_fn = nn.CrossEntropyLoss().cuda()

            average_q_px_dist_per_head_per_block(val_path.split('/')[-1], val_path, loader, model)
            clean_metrics = validate(model, loader, validate_loss_fn, val_path)
            adv_metrics = validate_attack(model, loader, validate_loss_fn, val_path)
            freq_hist(val_path.split('/')[-1], val_path)

            with open(val_path + '/Validation.csv', 'w+', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['Data', 'Loss', 'Acc@1', 'Acc@5'])
                writer.writerow(['Clean', clean_metrics['loss'], clean_metrics['top1'], clean_metrics['top5']])
                writer.writerow(['Adv', adv_metrics['loss'], adv_metrics['top1'], adv_metrics['top5']])
        else:
            print("Error: Results already existing:", val_path.split('/')[-1])
        print("\tUpdating CKA and adversarial CKA visualizations regardless.")
        loss_fn = nn.CrossEntropyLoss().cuda()
        ckpt_file = ckpt_path + ext
        if args.p and not custom_model:
            model = timm.create_model(tested_models[args.version], pretrained=True)
        else:
            model = timm.create_model(
                'custom_' + tested_models[args.version] if custom_model else tested_models[args.version],
                checkpoint_path=ckpt_file)
        model = model.cuda()
        for model_name in tested_models:
            for version in custom_versions:
                ckpt_file = train_path + model_name + '_' + version + ext
                if os.path.exists(ckpt_file):
                    get_CKA(val_path, model, exp_name, timm.create_model('custom_' + model_name, checkpoint_path=ckpt_file).cuda(), 'custom_' + model_name+'_'+version, loader)
                    get_adversarial_CKA(val_path, model, exp_name,
                                        timm.create_model('custom_' + model_name, checkpoint_path=ckpt_file).cuda(),
                                        'custom_' + model_name + '_' + version, loader, loss_fn)
            ckpt_file = train_path + model_name + '_scratch' + ext
            if os.path.exists(ckpt_file):
                get_CKA(val_path, model, exp_name, timm.create_model(model_name, checkpoint_path=ckpt_file).cuda(), model_name+'_scratch', loader)
                get_adversarial_CKA(val_path, model, exp_name, timm.create_model(model_name, checkpoint_path=ckpt_file).cuda(),
                                    model_name + '_scratch', loader, loss_fn)
            get_CKA(val_path, model, exp_name, timm.create_model(model_name, pretrained=True).cuda(), model_name+'_pretrained', loader)
            get_adversarial_CKA(val_path, model, exp_name, timm.create_model(model_name, pretrained=True).cuda(),
                                model_name + '_pretrained', loader, loss_fn)
    else:
        print("Error: Model asked does not exist:", ckpt_path.split('/')[-1])
        if not custom_model:
            print("\tOnly the pretrained version is available.")


if __name__ == '__main__':
    main()
