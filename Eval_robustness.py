from torch.nn import CrossEntropyLoss
from utils_eval import *
import argparse
import warnings
import models
import torch
import json
import timm
import os


parser = argparse.ArgumentParser()
parser.add_argument('-data', default='/home2/hugo/ImageNet', type=str)
parser.add_argument('-version', default=0, type=int)
parser.add_argument('-ckpt', default='', type=str)
parser.add_argument('-p', action='store_true', default=False)
parser.add_argument('-b', default=64, type=int)
parser.add_argument('-epsilon', default=0.062, type=float)
parser.add_argument('-steps', default=1, type=int)
parser.add_argument('-step_size', default=1, type=float)
parser.add_argument('-gpu', default=0, type=int)  # 0 for the RTX A5000 on workstation
parser.add_argument('-CKA', action='store_true', default=False)
parser.add_argument('-CKA_single', action='store_true', default=False)
parser.add_argument('-all_exp', action='store_true', default=False)


def main():
    args = parser.parse_args()
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
    else:
        torch.cuda.set_device(0)
    assert args.rank >= 0
    if args.steps != 1:
        args.step_size = 0.025
    tested_models = ['vit_base_patch16_224', 'vit_base_patch32_224', 't2t_vit_14']
    vit_versions = ['doexp5', 'donegexp025l', 'donegexp025l_finetuned']
    t2t_versions = ['t', 'p', 't_doexp05l', 't_donegexp075l', 't_donegexp05l', 't_donegexp025l']
    if args.all_exp:
        tested_models = tested_models + ['vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_small_patch32_224']
        vit_versions = vit_versions + ['dosq4015']
    train_path = 'output/train/'
    val_path = 'output/val/'
    ext = '/model_best.pth.tar'
    if os.path.exists(val_path + 'all_summaries.json'):
        with open(val_path + 'all_summaries.json', 'r') as json_file:
            all_summaries = json.load(json_file)
    else:
        all_summaries = {}
        with open(val_path + 'all_summaries.json', 'w+') as f:
            json.dump(all_summaries, f)
    json_file = val_path + 'all_summaries.json'
    if args.version < 0 or args.version >= len(tested_models):
        if args.local_rank == 0:
            print("Error: Version asked does not exist.")
        return

    if 't2t' not in tested_models[args.version]:
        if args.ckpt in vit_versions:
            ckpt_path = train_path + tested_models[args.version] + '_' + args.ckpt
            exp_name = tested_models[args.version] + '_' + args.ckpt
            model_name = 'custom_' + tested_models[args.version]
        else:
            ckpt_path = train_path + tested_models[args.version]
            exp_name = tested_models[args.version] + ('_pretrained' if args.p else '_scratch')
            model_name = tested_models[args.version]
    elif args.ckpt in t2t_versions:
        ckpt_path = train_path + tested_models[args.version] + '_' + args.ckpt
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
    if args.local_rank == 0:
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
    if args.distributed:
        model = NativeDDP(model, device_ids=[args.local_rank])
    if exp_name not in all_summaries.keys():
        all_summaries[exp_name] = {}
    if os.path.exists(ckpt_path) or args.p:
        loader = get_val_loader(args.data, batch_size=args.b)
        loss_fn = CrossEntropyLoss().cuda()

        attn_distance(model, exp_name, loader, all_summaries[exp_name], args)
        save_experiment_results(json_file, all_summaries, args.local_rank)
        adv_attn_distance(model, exp_name, loss_fn, loader, all_summaries[exp_name], args, epsilonMax=args.epsilon, pgd_steps=args.steps, step_size=args.step_size)
        save_experiment_results(json_file, all_summaries, args.local_rank)
        validate(model, loader, loss_fn, all_summaries[exp_name], args)
        save_experiment_results(json_file, all_summaries, args.local_rank)
        validate_attack(model, loader, loss_fn, all_summaries[exp_name], args, epsilonMax=args.epsilon, pgd_steps=args.steps, step_size=args.step_size)
        save_experiment_results(json_file, all_summaries, args.local_rank)

        if args.CKA:
            # do_all_CKAs(get_CKAs, all_summaries, json_file, model, exp_name, loader, loss_fn, tested_models,
            #             vit_versions, t2t_versions, train_path, ext, args)
            pass
        if args.CKA_single or args.all_exp:
            # do_all_CKAs(get_CKAs_single_element, all_summaries, json_file, model, exp_name, loader, loss_fn,
            #             tested_models, vit_versions, t2t_versions, train_path, ext, args)
            pass


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()
