from utils_eval import attn_distance, adv_attn_distance, freq_hist, get_CKAs
from torchvision.utils import save_image
from collections import OrderedDict
from contextlib import suppress
from timm.utils import *
from torch import nn
import argparse
import warnings
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


def save_experiment_results(json_file, data):
    with open(json_file, 'w') as f:
        json.dump(data, f)

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


def validate(model, loader, loss_fn, val_path, summary):
    if 'Metrics_cln' in summary.keys():
        return summary['Metrics_cln']
    print('\t---Starting validation on clean DS---')
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

    summary['Metrics_cln'] = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])
    save_image((cleanImage[:16] + 1) / 2, val_path + '/clean_batch.png')

    return summary['Metrics_cln']


def validate_attack(model, loader, loss_fn, val_path, summary, epsilonMax=0.062):
    if 'Metrics_adv' in summary.keys():
        return summary['Metrics_adv']
    print('\t---Starting validation on attacked DS---')
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

    summary['Metrics_adv'] = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])
    save_image((perturbedImage[:16] + 1) / 2, val_path + '/adv_batch.png')
    save_image(perturbationVal[:16], val_path + '/perturb_batch.png')

    return summary['Metrics_adv']


def main():
    tested_models = ['vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_small_patch32_224', 'vit_base_patch16_224',
                     'vit_base_patch32_224', 't2t_vit_14']
    vit_versions = ['doexp5', 'dosq4015']
    t2t_versions = ['t', 'p', 't_doexp05l']
    train_path = 'output/train/'
    val_path = 'output/val/'
    ext = '/model_best.pth.tar'
    args = parser.parse_args()
    if os.path.exists(val_path + 'all_summaries.json'):
        with open(val_path + 'all_summaries.json', 'r') as json_file:
            all_summaries = json.load(json_file)
    else:
        all_summaries = {}
        with open(val_path + 'all_summaries.json', 'w+') as f:
            json.dump(all_summaries, f)
    json_file = val_path + 'all_summaries.json'
    if args.version < 0 or args.version >= len(tested_models):
        print("Error: Version asked does not exist.")
        return

    if 't2t' not in tested_models[args.version]:
        if args.ckpt in vit_versions:
            ckpt_path = train_path + tested_models[args.version] + '_' + args.ckpt
            val_path = val_path + tested_models[args.version] + '_' + args.ckpt
            exp_name = tested_models[args.version] + '_' + args.ckpt
            model_name = 'custom_' + tested_models[args.version]
        else:
            ckpt_path = train_path + tested_models[args.version]
            val_path = val_path + tested_models[args.version] + ('_pretrained' if args.p else '_scratch')
            exp_name = tested_models[args.version] + ('_pretrained' if args.p else '_scratch')
            model_name = tested_models[args.version]
    elif args.ckpt in t2t_versions:
        ckpt_path = train_path + tested_models[args.version] + '_' + args.ckpt
        val_path = val_path + tested_models[args.version] + '_' + args.ckpt
        exp_name = tested_models[args.version] + '_' + args.ckpt
        if args.ckpt in ['t', 'p']:
            model_name = tested_models[args.version] + '_' + args.ckpt
        else:
            model_name = 'custom_' + tested_models[args.version] + '_' + args.ckpt.split('_')[0]
    else:
        return

    ckpt_file = ckpt_path + ext
    if not args.p and not os.path.exists(ckpt_file):
        return
    print('\n\t======Starting evaluation of ' + exp_name + '======')
    if 't2t' not in model_name:
        if args.p:
            model = timm.create_model(model_name, pretrained=True)
        else:
            model = timm.create_model(model_name, checkpoint_path=ckpt_file)
    else:
        if args.ckpt in ['t', 'p']:
            model = models.T2T.load_t2t_vit(model_name, ckpt_file)
        else:
            model = models.Custom_T2T.load_custom_t2t_vit(model_name, ckpt_file)
    model = model.cuda()
    if exp_name not in all_summaries.keys():
        all_summaries[exp_name] = {}
    if os.path.exists(ckpt_path) or args.p:
        loader = get_val_loader(args.data, batch_size=16)
        if not os.path.exists(val_path):
            os.mkdir(val_path)
        validate_loss_fn = nn.CrossEntropyLoss().cuda()

        attn_distance(val_path.split('/')[-1], val_path, loader, model, all_summaries[exp_name])
        save_experiment_results(json_file, all_summaries)
        adv_attn_distance(val_path.split('/')[-1], val_path, loader, model, validate_loss_fn, all_summaries[exp_name])
        save_experiment_results(json_file, all_summaries)
        validate(model, loader, validate_loss_fn, val_path, all_summaries[exp_name])
        save_experiment_results(json_file, all_summaries)
        validate_attack(model, loader, validate_loss_fn, val_path, all_summaries[exp_name])
        save_experiment_results(json_file, all_summaries)
        # freq_hist(val_path.split('/')[-1], val_path)

        loss_fn = nn.CrossEntropyLoss().cuda()
        loader = get_val_loader(args.data, batch_size=args.b)
        for model_name in tested_models:
            if 't2t' not in model_name:
                for version in vit_versions:
                    ckpt_file = train_path + model_name + '_' + version + ext
                    if os.path.exists(ckpt_file):
                        get_CKAs(all_summaries[exp_name], model, 'custom_' + model_name, model_name + '_' + version, loader, loss_fn, model_2_ckpt_file=ckpt_file)
                        save_experiment_results(json_file, all_summaries)
                ckpt_file = train_path + model_name + ext
                if os.path.exists(ckpt_file):
                    get_CKAs(all_summaries[exp_name], model, model_name, model_name + '_scratch', loader, loss_fn, model_2_ckpt_file=ckpt_file)
                    save_experiment_results(json_file, all_summaries)
                get_CKAs(all_summaries[exp_name], model, model_name, model_name + '_pretrained', loader, loss_fn, pretrained=True)
                save_experiment_results(json_file, all_summaries)
            else:
                for version in t2t_versions:
                    ckpt_file = train_path + model_name + '_' + version + ext
                    if os.path.exists(ckpt_file):
                        if version in ['p', 't']:
                            model_type = model_name + '_' + version
                        else:
                            model_type = 'custom_' + model_name + '_' + version
                        get_CKAs(all_summaries[exp_name], model, model_type, model_name + '_' + version, loader, loss_fn, model_2_ckpt_file=ckpt_file)
                        save_experiment_results(json_file, all_summaries)


if __name__ == '__main__':
    # print("\n/!\\ The warnings are disabled!")
    warnings.filterwarnings("ignore")
    torch.cuda.set_device(0)  # set GPU 0 for the RTX A5000 on workstation
    main()
