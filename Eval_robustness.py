from utils_eval import attn_distance, adv_attn_distance, freq_hist, CKA_in_summaries
from torchvision.utils import save_image
from collections import OrderedDict
from contextlib import suppress
from timm.utils import *
from torch import nn
import argparse
import models
import json
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
parser.add_argument('-b', default=32, type=int)


def get_val_loader(data_path, batch_size=64):
    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
    dataset_eval = timm.data.create_dataset('', root=data_path, split='validation', is_training=False, batch_size=batch_size)
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


def validate_attack(model, loader, loss_fn, val_path, epsilonMax=0.062):
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
    tested_models = ['vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_small_patch32_224', 'vit_base_patch16_224',
                     'vit_base_patch32_224', 't2t_vit_14']
    custom_vit_versions = ['doexp5', 'dosq4015']
    t2t_versions = ['t', 'p']
    args = parser.parse_args()
    custom_vit = args.ckpt if args.ckpt in custom_vit_versions else None
    t2t_vit_mode = args.ckpt if args.ckpt in t2t_versions else None
    train_path = 'output/train/'
    val_path = 'output/val/'
    ext = '/model_best.pth.tar'
    if args.version < 0 or args.version >= len(tested_models):
        print("Error: Version asked does not exist.")
        return
    if custom_vit is not None:
        ckpt_path = train_path + tested_models[args.version] + '_' + custom_vit
        val_path = val_path + tested_models[args.version] + '_' + custom_vit
        exp_name = 'custom_' + tested_models[args.version] + '_' + custom_vit
        model_name = 'custom_' + tested_models[args.version]
    elif t2t_vit_mode is not None:
        ckpt_path = train_path + tested_models[args.version] + '_' + t2t_vit_mode
        val_path = val_path + tested_models[args.version] + '_' + t2t_vit_mode
        exp_name = tested_models[args.version] + '_' + t2t_vit_mode
        model_name = tested_models[args.version] + '_' + t2t_vit_mode
    else:
        ckpt_path = train_path + tested_models[args.version]
        val_path = val_path + tested_models[args.version] + ('_pretrained' if args.p else '_scratch')
        exp_name = tested_models[args.version] + ('_pretrained' if args.p else '_scratch')
        model_name = tested_models[args.version]
    ckpt_file = ckpt_path + ext
    if t2t_vit_mode:
        model = models.T2T.load_t2t_vit(model_name, ckpt_file)
    elif not args.p:
        model = timm.create_model(model_name, checkpoint_path=ckpt_file)
    else:
        model = timm.create_model(model_name, pretrained=True)
    model = model.cuda()
    json_summaries = {}
    if os.path.exists(ckpt_path) or args.p:
        if not os.path.exists(val_path):
            loader = get_val_loader(args.data, batch_size=128)
            os.mkdir(val_path)
            validate_loss_fn = nn.CrossEntropyLoss().cuda()

            att_distances = attn_distance(val_path.split('/')[-1], val_path, loader, model)
            adv_att_distances = adv_attn_distance(val_path.split('/')[-1], val_path, loader, model, validate_loss_fn)
            clean_metrics = validate(model, loader, validate_loss_fn, val_path)
            adv_metrics = validate_attack(model, loader, validate_loss_fn, val_path)
            freq_hist(val_path.split('/')[-1], val_path)

            json_summaries['att_distances'] = att_distances.tolist()
            json_summaries['adv_att_distances'] = adv_att_distances.tolist()
            json_summaries['clean_metrics'] = clean_metrics
            json_summaries['adv_metrics'] = adv_metrics

            with open(val_path + '/Validation.csv', 'w+', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['Data', 'Loss', 'Acc@1', 'Acc@5'])
                writer.writerow(['Clean', clean_metrics['loss'], clean_metrics['top1'], clean_metrics['top5']])
                writer.writerow(['Adv', adv_metrics['loss'], adv_metrics['top1'], adv_metrics['top5']])
        else:
            with open(val_path + '/json_summaries.json', 'r') as j_file:
                json_summaries = json.load(j_file)
        loss_fn = nn.CrossEntropyLoss().cuda()
        loader = get_val_loader(args.data, batch_size=args.b)
        for model_name in tested_models:
            for version in custom_vit_versions:
                ckpt_file = train_path + model_name + '_' + version + ext
                if os.path.exists(ckpt_file):
                    json_summaries = CKA_in_summaries(val_path, model, exp_name, 'custom_' + model_name, 'custom_' + model_name+'_'+version, loader, loss_fn, json_summaries, model_2_ckpt_file=ckpt_file)
            for version in t2t_versions:
                ckpt_file = train_path + model_name + '_' + version + ext
                if os.path.exists(ckpt_file):
                    json_summaries = CKA_in_summaries(val_path, model, exp_name, model_name + '_' + version, model_name + '_' + version, loader, loss_fn, json_summaries, model_2_ckpt_file=ckpt_file)
            ckpt_file = train_path + model_name + ext
            if os.path.exists(ckpt_file):
                json_summaries = CKA_in_summaries(val_path, model, exp_name, model_name, model_name + '_scratch', loader, loss_fn, json_summaries, model_2_ckpt_file=ckpt_file)
            json_summaries = CKA_in_summaries(val_path, model, exp_name, model_name, model_name + '_pretrained', loader, loss_fn, json_summaries, pretrained=True)
        with open(val_path + '/json_summaries.json', 'w+') as j_file:
            json.dump(json_summaries, j_file)


if __name__ == '__main__':
    torch.cuda.set_device(1)
    main()
