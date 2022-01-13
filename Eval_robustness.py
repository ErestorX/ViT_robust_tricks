from utils_eval import *
from torch import nn
import argparse
import warnings
import models
import json
import torch
import timm
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
        loader = get_val_loader(args.data, batch_size=64)
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
        t2t_model_1 = 't2t' in model_name
        for model_name in tested_models:
            if 't2t' not in model_name:
                for version in vit_versions:
                    ckpt_file = train_path + model_name + '_' + version + ext
                    if os.path.exists(ckpt_file):
                        get_CKAs(all_summaries[exp_name], model, 'custom_' + model_name, model_name + '_' + version, loader, loss_fn, model_2_ckpt_file=ckpt_file, t2t_model_1=t2t_model_1)
                        save_experiment_results(json_file, all_summaries)
                ckpt_file = train_path + model_name + ext
                if os.path.exists(ckpt_file):
                    get_CKAs(all_summaries[exp_name], model, model_name, model_name + '_scratch', loader, loss_fn, model_2_ckpt_file=ckpt_file, t2t_model_1=t2t_model_1)
                    save_experiment_results(json_file, all_summaries)
                get_CKAs(all_summaries[exp_name], model, model_name, model_name + '_pretrained', loader, loss_fn, pretrained=True, t2t_model_1=t2t_model_1)
                save_experiment_results(json_file, all_summaries)
            else:
                for version in t2t_versions:
                    ckpt_file = train_path + model_name + '_' + version + ext
                    if os.path.exists(ckpt_file):
                        if version in ['p', 't']:
                            model_type = model_name + '_' + version
                        else:
                            model_type = 'custom_' + model_name + '_' + '_'.join(version.split('_')[:-1])
                        get_CKAs(all_summaries[exp_name], model, model_type, model_name + '_' + version, loader, loss_fn, model_2_ckpt_file=ckpt_file, t2t_model_1=t2t_model_1)
                        save_experiment_results(json_file, all_summaries)


if __name__ == '__main__':
    # print("\n/!\\ The warnings are disabled!")
    warnings.filterwarnings("ignore")
    torch.cuda.set_device(0)  # set GPU 0 for the RTX A5000 on workstation
    main()
