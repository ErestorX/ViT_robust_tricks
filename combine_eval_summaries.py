import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import numpy as np
import argparse
from tabulate import tabulate
import json
import os


parser = argparse.ArgumentParser()
parser.add_argument('--val_path', default='./output/val/', type=str)


def order_exp(val_path, exp_list):
    models_list = ['_'.join(model.split('_')[:-1]) for model in exp_list]
    models_list = sorted(list(set(models_list)))
    id_exp = ['_'.join(model.split('_')[-1:]) for model in exp_list]
    id_exp = sorted(list(set(id_exp)))
    id_exp.remove('pretrained')
    id_exp.remove('scratch')
    id_exp = ['pretrained', 'scratch'] + id_exp
    exp_list = []
    for model_name in models_list:
        for exp_name in id_exp:
            if os.path.exists(val_path + model_name + '_' + exp_name):
                exp_list.append('_'.join([model_name, exp_name]))
    return exp_list


def compare_att_distances_model(data, model, attacks):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    list_exp = [data[model]['AttDist_cln']]
    legends = ['Clean']
    for attack in attacks:
        if 'AttDist_adv' + attack in data[model]:
            list_exp.append(data[model]['AttDist_adv' + attack])
            param = attack.split('_')[1:]
            param = [float(p.split(':')[-1]) for p in param]
            steps, eps = param[0], param[1]
            title = ('FGSM' if steps == 1 else 'PGD') + ' ' + str(eps)
            legends.append(title)
    nb_exp = len(list_exp)
    colors = colors[:nb_exp]
    list_blocks = []
    block_width = 0.8/nb_exp
    if 't2t' in model:
        for i in range(nb_exp):
            list_blocks.append(np.arange(-2+(i*block_width), len(list_exp[i])-2+(i*block_width), 1.0))
    else:
        for i in range(nb_exp):
            list_blocks.append(np.arange(0+(i*block_width), len(list_exp[i])+(i*block_width), 1.0))
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.title('Attention distance on ' + model)
    list_bp = [ax.boxplot(e, positions=b, patch_artist=True, widths=block_width) for e, b in zip(list_exp, list_blocks)]
    for bp, color in zip(list_bp, colors):
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color=color)
        for patch in bp['boxes']:
            patch.set(facecolor=color)
    ax.legend([bp["boxes"][0] for bp in list_bp], legends)
    ax.set_ylim(0, 224)
    ax.yaxis.grid(True)
    list_blocks = [np.round(b-(block_width*id), 0).astype(np.int).tolist() for id, b in enumerate(list_blocks)]
    blocks = []
    for b in list_blocks:
        blocks += b
    blocks = sorted(list(set(blocks)))
    ax.set_xticks(blocks)
    if 't2t' in model:
        plt.axvline(x=-.1, color='grey', alpha=.8)
    plt.savefig('output/val/AttDist_' + model + '.png')


def compare_att_distances_attack(data, attack, models):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    list_exp =[]
    legends = []
    for model in models:
        if attack in data[model]:
            list_exp.append(data[model][attack])
            legends.append(model)
    nb_exp = len(list_exp)
    colors = colors[:nb_exp]
    list_blocks = []
    block_width = 0.8 / nb_exp
    t2t = False
    for i in range(nb_exp):
        if 't2t' in legends[i]:
            t2t = True
            list_blocks.append(np.arange(-2 + (i * block_width), len(list_exp[i]) - 2 + (i * block_width), 1.0))
        else:
            list_blocks.append(np.arange(0 + (i * block_width), len(list_exp[i]) + (i * block_width), 1.0))
    fig, ax = plt.subplots(figsize=(10, 5))
    if attack == 'AttDist_cln':
        title = 'Clean data'
    else:
        param = attack.split('_')[-2:]
        param = [float(p.split(':')[-1]) for p in param]
        steps, eps = param[0], param[1]
        title = ('FGSM' if steps == 1 else 'PGD') + '_' + str(eps)
    plt.title('Attention distance on ' + title)
    list_bp = [ax.boxplot(e, positions=b, patch_artist=True, widths=block_width) for e, b in zip(list_exp, list_blocks)]
    for bp, color in zip(list_bp, colors):
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color=color)
        for patch in bp['boxes']:
            patch.set(facecolor=color)
    ax.legend([bp["boxes"][0] for bp in list_bp], legends)
    ax.set_ylim(0, 224)
    ax.yaxis.grid(True)
    list_blocks = [np.round(b - (block_width * id), 0).astype(np.int).tolist() for id, b in enumerate(list_blocks)]
    blocks = []
    for b in list_blocks:
        blocks += b
    blocks = sorted(list(set(blocks)))
    ax.set_xticks(blocks)
    if t2t:
        plt.axvline(x=-.1, color='grey', alpha=.8)
    plt.savefig('output/val/AttDist_' + title + '.png')


def get_top1_val(data):
    list_models = ['t2t_vit_14_p', 't2t_vit_14_t', 't2t_vit_14_t_doexp05l', 't2t_vit_14_t_donegexp05l', 'vit_base_patch16_224_pretrained', 'vit_base_patch32_224_pretrained', 'vit_base_patch32_224_scratch', 'vit_base_patch32_224_doexp5']
    attacks = ['_steps:40_eps:0.001', '_steps:40_eps:0.003', '_steps:40_eps:0.005', '_steps:40_eps:0.01',
               '_steps:1_eps:0.031', '_steps:1_eps:0.062']
    list_top1 = [[data[exp]['Metrics_cln']['top1'] for exp in list_models]]
    attack_names = ['Clean data']
    for a in attacks:
        list_top1.append([data[exp]['Metrics_adv'+a]['top1'] for exp in list_models])
        attack_name = [p.split(':')[1] for p in a.split('_')[1:]]
        attack_names.append(('FGSM' if attack_name[0] == '1' else 'PGD') + '_' + attack_name[1])
    list_top1 = np.asarray(list_top1).swapaxes(0, 1)
    list_top1 = list_top1.tolist()
    table = [[exp] + val for exp, val in zip(list_models, list_top1)]
    print(tabulate(table, headers=['Model']+attack_names))


def main():
    data = json.load(open('saves/all_summaries_01-11_17:00.json', 'r'))
    get_top1_val(data)
    exp = list(data.keys())
    for e1 in exp:
        compare_att_distances_model(data, e1)
        for e2 in exp:
            compare_att_distances_attack(data, e1, e2)
            compare_att_distances_attack(data, e1, e2, adv_ds='AttDist_trf_steps:1_eps:0.062')


if __name__ == '__main__':
    # main()
    data = json.load(open('saves/all_summaries_01-19_10:30.json', 'r'))
    get_top1_val(data)
