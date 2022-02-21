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
    FGSM_blue = ['midnightblue', 'mediumblue', 'blue', 'mediumslateblue', 'darkorchid', 'fushia', 'violet', 'hotpink', 'pink']
    blue_id = 0
    PGD_green = ['darkgreen', 'green', 'limegreen', 'mediumseagreen', 'aqquamarine', 'turquoise', 'paleturquoise', 'lightseagreen', 'darkcyan']
    green_id = 0
    colors = []
    list_exp = [data[model]['AttDist_cln']]
    legends = ['Clean']
    for attack in attacks:
        if 'AttDist_adv' + attack in data[model]:
            list_exp.append(data[model]['AttDist_adv' + attack])
            param = attack.split('_')[1:]
            param = [float(p.split(':')[-1]) for p in param]
            steps, eps = param[0], param[1]
            if steps == 1:
                colors.append(FGSM_blue[blue_id])
                blue_id += 1
            else:
                colors.append(PGD_green[green_id])
                green_id += 1
            title = ('FGSM' if steps == 1 else 'PGD') + ' ' + str(eps)
            legends.append(title)
    nb_exp = len(list_exp)
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
    ax.set_xlabel('Block ID')
    ax.set_ylabel('Attention distance, in pixels')
    if 't2t' in model:
        plt.axvline(x=-.1, color='grey', alpha=.8)
        ax.set_xlabel('Block ID, negative for the T2T blocks')
    plt.savefig('output/val/plots/AttDist_' + model + '.png')


def compare_att_distances_attack(data, attack, models):
    t2t_blue = ['midnightblue', 'mediumblue', 'blue', 'mediumslateblue', 'darkorchid', 'fushia', 'violet', 'hotpink',
                 'pink']
    blue_id = 0
    vit_green = ['darkgreen', 'green', 'limegreen', 'mediumseagreen', 'aquamarine', 'turquoise', 'paleturquoise',
                 'lightseagreen', 'darkcyan']
    green_id = 0
    colors = []
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
            colors.append(t2t_blue[blue_id])
            blue_id += 1
        else:
            list_blocks.append(np.arange(0 + (i * block_width), len(list_exp[i]) + (i * block_width), 1.0))
            colors.append(vit_green[green_id])
            green_id += 1
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
    ax.set_xlabel('Block ID')
    ax.set_ylabel('Attention distance, in pixels')
    if t2t:
        plt.axvline(x=-.1, color='grey', alpha=.8)
    ax.set_xlabel('Block ID, negative for the T2T blocks')
    plt.savefig('output/val/plots/AttDist_' + title + '.png')


def get_top1_val(data, experiments, model_list):
    t2t_blue = ['midnightblue', 'mediumblue', 'blue', 'mediumslateblue', 'darkorchid', 'fuchsia', 'violet', 'hotpink',
                'pink']
    t2t_blue.reverse()
    blue_id = 0
    vit_green = ['darkgreen', 'green', 'limegreen', 'mediumseagreen', 'aquamarine', 'turquoise', 'paleturquoise',
                 'lightseagreen', 'darkcyan']
    green_id = 0
    if model_list is None:
        model_list = list(data.keys())
    cln = np.asarray([data[model]['Metrics_cln']['top1'] for model in model_list])
    exp_title = ['Clean']
    per_model_evol = np.reshape(cln, (cln.shape[0], 1))
    for experiment in experiments:
        param = experiment.split('_')[1:]
        param = [float(p.split(':')[-1]) for p in param]
        steps, eps = param[0], param[1]
        title = ('FGSM' if steps == 1 else 'PGD') + ' ' + str(eps)
        exp_title.append(title)
        experiment = 'Metrics_adv' + experiment
        adv = np.asarray([data[model][experiment]['top1'] for model in model_list])
        adv = adv.reshape((adv.shape[0], 1))
        per_model_evol = np.concatenate((per_model_evol, adv), axis=1)
    for i in range(per_model_evol.shape[0]):
        if 't2t' in model_list[i]:
            color = t2t_blue[blue_id]
            blue_id += 1
        else:
            color = vit_green[green_id]
            green_id += 1
        plt.plot(exp_title, per_model_evol[i], label=model_list[i], color=color)
    plt.axvline(x=exp_title[1], color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=exp_title[5], color='gray', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('output/val/plots/Clean_vs_Adversarial_acc.png')


def AttDist_vs_top1(data, attack, list_models):
    t2t_blue = ['midnightblue', 'mediumblue', 'blue', 'mediumslateblue', 'darkorchid', 'fuchsia', 'violet', 'hotpink',
                'pink']
    t2t_blue.reverse()
    blue_id = 0
    vit_green = ['darkgreen', 'green', 'limegreen', 'mediumseagreen', 'aquamarine', 'turquoise', 'paleturquoise',
                 'lightseagreen', 'darkcyan']
    green_id = 0
    t2t_block_ckpt = [2, 4, 5, 7, 9, 10, -1]
    vit_block_ckpt = [0, 2, 3, 5, 7, 8, -1]
    colors = []
    top1 = []
    AttDist = []
    for model in list_models:
        top1.append(data[model]['Metrics'+attack]['top1'])
        dist = data[model]['AttDist' + attack]
        AttDist.append(dist)
        if 't2t' in model:
            colors.append(t2t_blue[blue_id])
            blue_id += 1
        else:
            colors.append(vit_green[green_id])
            green_id += 1
    data_figs = {'block_t2t':{}, 'final_block':{}}
    for id in vit_block_ckpt[:-1]:
        data_figs['block_'+str(id)] = {}
    for model, acc, dist in zip(list_models, top1, AttDist):
        if 't2t' in model:
            data_figs['block_t2t'][model] = [acc, dist[0]]
            for block_id in t2t_block_ckpt:
                if block_id == -1:
                    data_figs['final_block'][model] = [acc, dist[block_id]]
                else:
                    data_figs['block_'+str(block_id-2)][model] = [acc, dist[block_id]]
        else:
            for block_id in vit_block_ckpt:
                if block_id == -1:
                    data_figs['final_block'][model] = [acc, dist[block_id]]
                else:
                    data_figs['block_'+str(block_id)][model] = [acc, dist[block_id]]
    for block in data_figs.keys():
        fig, ax = plt.subplots(figsize=(10, 5))
        if attack == '_cln':
            type_attack = 'Clean'
        else:
            params = [x.split(':')[1] for x in attack.split('_')[2:]]
            type_attack = ('FGSM' if params[0] == '1' else 'PGD') + '_' + params[1]
        plt.title('Accuracy '+type_attack+' vs Attention distance on ' + block)
        acc = []
        distance_points = []
        legends = []
        if block == 'block_t2t':
            for model in data_figs[block].keys():
                acc.append(round(data_figs[block][model][0], 2))
                distance_points.append(data_figs[block][model][1])
                legends.append(model)
            bp = ax.boxplot(distance_points, positions=acc, vert=False, patch_artist=True, widths=1.5)
            for patch, color in zip(bp['boxes'], t2t_blue[:blue_id]):
                patch.set(facecolor=color)
        else:
            for model in data_figs[block].keys():
                acc.append(round(data_figs[block][model][0], 2))
                distance_points.append(data_figs[block][model][1])
                legends.append(model)
            bp = ax.boxplot(distance_points, positions=acc, vert=False, patch_artist=True, widths=1.5)
            for patch, color in zip(bp['boxes'], colors):
                patch.set(facecolor=color)
        ax.legend(bp['boxes'], legends, loc='lower right')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Attention distance, in pixels')
        ax.set_yticks(acc + np.arange(0, 101, 10).tolist(), acc + np.arange(0, 101, 10).tolist())
        ax.set_xticks(np.arange(0, 226, 25))
        plt.tight_layout()
        plt.savefig('output/val/plots/Acc_'+type_attack+'_vs_AttDist_' + block + '.png')


def main():
    data = json.load(open('output/val/all_summaries.json', 'r'))
    attacks = ['_steps:40_eps:0.001', '_steps:40_eps:0.003', '_steps:40_eps:0.005', '_steps:40_eps:0.01',
               '_steps:1_eps:0.031', '_steps:1_eps:0.062']
    list_models = ['t2t_vit_14_p', 't2t_vit_14_t', 't2t_vit_14_t_doexp05l', 't2t_vit_14_t_donegexp05l', 't2t_vit_14_t_donegexp025l',
                   'vit_base_patch16_224_pretrained', 'vit_base_patch32_224_pretrained', 'vit_base_patch32_224_scratch',
                   'vit_base_patch32_224_doexp5', 'vit_base_patch32_224_donegexp025l']
    get_top1_val(data, attacks, list_models)



if __name__ == '__main__':
    main()
