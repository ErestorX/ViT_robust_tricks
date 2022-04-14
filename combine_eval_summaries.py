import matplotlib.pyplot as plt
import numpy as np
import argparse
from tabulate import tabulate
import json
import os


parser = argparse.ArgumentParser()
parser.add_argument('--val_path', default='./output/val/', type=str)


def beautiful_model_name(model_name):
    final_tokens = []
    for token in model_name.split("_"):
        if token == 't2t':
            final_tokens.append('T2T-')
        elif token == 'vit':
            final_tokens.append('ViT')
        elif token == 'base':
            final_tokens.append('-B')
        elif token == 'small':
            final_tokens.append('-S')
        elif token == 'tiny':
            final_tokens.append('-T')
        elif token == '14':
            pass
        elif token == 'patch16':
            final_tokens.append('/16')
        elif token == 'patch32':
            final_tokens.append('/32')
        elif token == '224':
            pass
        elif token == 'p':
            final_tokens.append('-p')
        elif token == 't':
            final_tokens.append('-t')
        elif token == 'pretrained':
            final_tokens.append(' Pretrained')
            # pass
        elif token == 'scratch':
            final_tokens.append(' INet training')
        elif token == 'doexp05l':
            final_tokens.append(' - dropout: 0.5/exp(0.5*l*d)')
        elif token == 'donegexp05l':
            final_tokens.append(' - dropout: 0.5-0.5/exp(0.5*l*d)')
        elif token == 'donegexp025l':
            final_tokens.append(' - dropout: 0.5-0.5/exp(0.25*l*d)')
        elif token == 'donegexp075l':
            final_tokens.append(' - dropout: 0.5-0.5/exp(0.75*l*d)')
        elif token == 'doexp5':
            final_tokens.append(' - dropout: 0.5/exp(5*d)')
        elif token == 'finetuned':
            final_tokens.append(' for INet fine tuning')
        else:
            final_tokens.append('-'+token)
    return ''.join(final_tokens)


def beautiful_attack_name(attack_name, cln_is_no=False):
    if attack_name == '_cln':
        return 'no' if cln_is_no else 'Clean'
    else:
        param = attack_name.split('_')[-2:]
        param = [float(p.split(':')[-1]) for p in param]
        steps, eps = param[0], param[1]
        atk_name = ('FGSM' if steps == 1 else 'PGD') + ' ' + str(eps)
        return atk_name

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


def table_top1_val(data, attacks, list_models):
    table = []
    headers = ['Model']
    for atk in attacks:
        if atk == '_cln':
            headers.append('Clean')
        else:
            param = atk.split('_')[1:]
            param = [float(p.split(':')[-1]) for p in param]
            steps, eps = param[0], param[1]
            atk_name = ('FGSM' if steps == 1 else 'PGD') + ' ' + str(eps)
            headers.append(atk_name)
    for model in list_models:
        row = [beautiful_model_name(model)]
        for attack in attacks:
            if attack == '_cln':
                row.append(data[model]['Metrics' + attack]['top1'])
            else:
                row.append(data[model]['Metrics_adv' + attack]['top1'])
        table.append(row)
    print(tabulate(table, headers=headers, tablefmt='latex'))


def compare_att_distances_model(data, model, attacks):
    FGSM_blue = ['midnightblue', 'mediumblue', 'blue', 'mediumslateblue', 'darkorchid', 'fuchsia', 'violet', 'hotpink']
    FGSM_blue.reverse()
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
    plt.title('Attention distance on ' + beautiful_model_name(model))
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


def compare_att_distances_model_avg(data, model, attacks):
    FGSM_blue = ['midnightblue', 'mediumblue', 'blue', 'mediumslateblue', 'darkorchid', 'fuchsia', 'violet', 'hotpink']
    FGSM_blue.reverse()
    blue_id = 0
    PGD_green = ['darkgreen', 'green', 'limegreen', 'mediumseagreen', 'aqquamarine', 'turquoise', 'paleturquoise',
                 'lightseagreen', 'darkcyan']
    green_id = 0
    colors = ['black']
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
    for i in range(nb_exp):
        if 't2t' in model:
            list_blocks.append(np.arange(-2, len(list_exp[i]) - 2, 1.0))
        else:
            list_blocks.append(np.arange(0, len(list_exp[i]), 1.0))
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.title('Attention distance on ' + beautiful_model_name(model))
    for id, e in enumerate(list_exp):
        avg_e = []
        for block in e:
            avg_e.append(np.mean(block))
        list_exp[id] = avg_e
    for blocks, e, color, legend in zip(list_blocks, list_exp, colors, legends):
        plt.plot(blocks, e, color=color, label=legend)
    ax.yaxis.grid(True)
    ax.set_xlabel('Block ID, negative for the T2T blocks')
    ax.set_ylabel('Attention distance, in pixels')
    if 't2t' in model:
        plt.axvline(x=-.5, color='grey', alpha=.8)
    plt.legend()
    plt.tight_layout()
    # ax.set_ylim(0, 224)
    plt.savefig('output/val/plots/AttDist_' + model + '_avg.png')


def compare_att_distances_attack(data, attack, models):
    t2t_blue = ['midnightblue', 'mediumblue', 'blue', 'mediumslateblue', 'darkorchid', 'fuchsia', 'violet', 'hotpink']
    t2t_blue.reverse()
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
            legends.append(beautiful_model_name(model))
    nb_exp = len(list_exp)
    colors = colors[:nb_exp]
    list_blocks = []
    block_width = 0.8 / nb_exp
    t2t = False
    for i in range(nb_exp):
        if 'T2T' in legends[i]:
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
    colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
              '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
              '#000075', '#a9a9a9', '#ffffff', '#000000']

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
    plt.tight_layout()
    plt.savefig('output/val/plots/AttDist_' + title + '_pretrainedModels.png')


def compare_att_distances_attack_avg(data, attack, models):
    t2t_blue = ['midnightblue', 'mediumblue', 'blue', 'mediumslateblue', 'darkorchid', 'fuchsia', 'violet', 'hotpink']
    t2t_blue.reverse()
    blue_id = 0
    vit_green = ['darkgreen', 'green', 'limegreen', 'mediumseagreen', 'aquamarine', 'turquoise', 'paleturquoise',
                 'lightseagreen', 'darkcyan']
    green_id = 0
    colors = []
    list_exp =[]
    legends = []
    list_top1 = []
    max_top1 = -1
    for model in models:
        if attack in data[model]:
            list_exp.append(data[model][attack])
            list_top1.append(data[model]['Metrics_'+'_'.join(attack.split('_')[1:])]['top1'])
            if data[model]['Metrics_'+'_'.join(attack.split('_')[1:])]['top1'] > max_top1:
                max_top1 = data[model]['Metrics_'+'_'.join(attack.split('_')[1:])]['top1']
            legends.append(beautiful_model_name(model) + '\nTop1: ')
    for id, (top1, legend) in enumerate(zip(list_top1, legends)):
        if top1 == max_top1:
            legends[id] = legend + "{:.2f}   -- Best --".format(top1)
        else:
            legends[id] = legend + "{:.2f}".format(top1)
    nb_exp = len(list_exp)
    colors = colors[:nb_exp]
    list_blocks = []
    t2t = False
    for i in range(nb_exp):
        if 't2t' in models[i]:
            t2t = True
            colors.append(t2t_blue[blue_id])
            blue_id += 1
            list_blocks.append(np.arange(-2, len(list_exp[i]) - 2, 1.0))
        else:
            colors.append(vit_green[green_id])
            green_id += 1
            list_blocks.append(np.arange(0, len(list_exp[i]), 1.0))
    fig, ax = plt.subplots(figsize=(10, 5))
    if attack == 'AttDist_cln':
        title = 'Clean data'
    else:
        param = attack.split('_')[-2:]
        param = [float(p.split(':')[-1]) for p in param]
        steps, eps = param[0], param[1]
        title = ('FGSM' if steps == 1 else 'PGD') + '_' + str(eps)
    plt.title('Attention distance on ' + title)
    for id, e in enumerate(list_exp):
        avg_e = []
        for block in e:
            avg_e.append(np.mean(block))
        list_exp[id] = avg_e
        colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#ffffff', '#000000']
    for blocks, e, color, legend in zip(list_blocks, list_exp, colors, legends):
        plt.plot(blocks, e, color=color, label=legend)
    # ax.set_ylim(0, 224)
    ax.yaxis.grid(True)
    ax.set_ylabel('Attention distance, in pixels')
    ax.set_xlabel('Block ID, negative for the T2T blocks')
    plt.tight_layout()
    ax.legend()
    if t2t:
        plt.axvline(x=-.5, color='grey', alpha=.8)
    plt.savefig('output/val/plots/AttDist_' + title + '_avg_pretrainedModels.png')


def get_top1_val(data, experiments, model_list):
    t2t_blue = ['midnightblue', 'mediumblue', 'blue', 'mediumslateblue', 'darkorchid', 'fuchsia', 'violet', 'hotpink']
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
        plt.plot(exp_title, per_model_evol[i], label=beautiful_model_name(model_list[i]), color=color)

    plt.axvspan(exp_title[1], exp_title[4], facecolor='grey', alpha=0.3)
    plt.legend()
    plt.ylabel('Top-1 accuracy')
    plt.title('Top-1 accuracy on clean and adversarial data')
    plt.tight_layout()
    plt.savefig('output/val/plots/Clean_vs_Adversarial_acc_baseModels.png')


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
                legends.append(beautiful_model_name(model))
            bp = ax.boxplot(distance_points, positions=acc, vert=False, patch_artist=True, widths=1.5)
            for patch, color in zip(bp['boxes'], t2t_blue[:blue_id]):
                patch.set(facecolor=color)
        else:
            for model in data_figs[block].keys():
                acc.append(round(data_figs[block][model][0], 2))
                distance_points.append(data_figs[block][model][1])
                legends.append(beautiful_model_name(model))
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


def CKA_and_attDist_plot(data, model_1, model_2, attack):
    pad = False
    if 't2t' in model_1 and 't2t' not in model_2 or 't2t' in model_2 and 't2t' not in model_1:
        pad = True
    if attack == '_cln':
        key = 'CKA_cln'
    else:
        key = 'CKA_trf' + attack
    try:
        CKA_model_1 = np.asarray(data[model_1][key][model_1])
        CKA_model_2 = np.asarray(data[model_2][key][model_2])
        if len(CKA_model_1) > 16:
            CKA_model_1 = CKA_model_1[3::3, 3::3]
        CKA_model_1 = (CKA_model_1 - np.min(CKA_model_1)) / (np.max(CKA_model_1) - np.min(CKA_model_1))
        if len(CKA_model_2) > 16:
            CKA_model_2 = CKA_model_2[3::3, 3::3]
        CKA_model_2 = (CKA_model_2 - np.min(CKA_model_2)) / (np.max(CKA_model_2) - np.min(CKA_model_2))
    except:
        print('CKA not available for ' + model_1 + ' or ' + model_2)
        return
    if attack == '_cln':
        key = 'AttDist_cln'
    else:
        key = 'AttDist_adv' + attack
    try:
        attDist_model1 = np.asarray(data[model_1][key])
        attDist_model2 = np.asarray(data[model_2][key])
    except:
        print('AttDist' + attack + ' not available for ' + model_1 + ' or ' + model_2)
        return

    if 't2t' not in model_1 and pad:
        CKA_model_1 = np.pad(CKA_model_1, ((2, 2), (2, 2)), 'constant', constant_values=0)
    if 't2t' not in model_2 and pad:
        CKA_model_2 = np.pad(CKA_model_2, ((2, 2), (2, 2)), 'constant', constant_values=0)
    avg = []
    for block in attDist_model1:
        avg.append(np.mean(block))
    attDist_model1 = np.asarray(avg)
    avg = []
    for block in attDist_model2:
        avg.append(np.mean(block))
    attDist_model2 = np.asarray(avg)
    pad_1 = 2 if 't2t' not in model_1 and pad else 0
    pad_2 = 2 if 't2t' not in model_2 and pad else 0
    blocks_model1 = list(range(pad_1, len(attDist_model1) + pad_1, 1))
    blocks_model2 = list(range(pad_2, len(attDist_model2) + pad_2, 1))
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
    axes[0].imshow(CKA_model_1, cmap='Reds')
    axes[0].set_title(beautiful_model_name(model_1) + '\nCKA')
    axes[0].set_xticks(range(0, len(CKA_model_1), 2))
    axes[0].set_yticks(range(0, len(CKA_model_1), 2))
    axes[0].set_xlabel('Blocks ID')
    axes[0].set_ylabel('Blocks ID')
    axes[2].imshow(CKA_model_2, cmap='Blues')
    axes[2].set_title(beautiful_model_name(model_2) + '\nCKA')
    axes[2].set_xticks(range(0, len(CKA_model_2), 2))
    axes[2].set_yticks(range(0, len(CKA_model_2), 2))
    axes[2].set_xlabel('Blocks ID')
    axes[2].set_ylabel('Blocks ID')
    axes[1].plot(attDist_model1, blocks_model1, color='red', label='Left model')
    axes[1].plot(attDist_model2, blocks_model2, color='blue', label='Right model')
    axes[1].set_ylim(axes[1].get_ylim()[::-1])
    axes[1].set_xlabel('Attention distance, in pixels')
    axes[1].set_ylabel('Blocks ID')
    min_x = min(int(np.min(attDist_model1)) - int(np.min(attDist_model1) % 10), int(np.min(attDist_model2)) - int(np.min(attDist_model2) % 10))
    max_x = max(int(np.max(attDist_model1)) + int(np.max(attDist_model1) % 10), int(np.max(attDist_model2)) + int(np.max(attDist_model2) % 10))
    axes[1].set_xlim([min_x, max_x])
    axes[1].set_xticks(range(min_x, max_x, 10))
    axes[1].set_yticks(range(0, int(np.max(blocks_model1)), 2))

    if attack == '_cln':
        attack_name = 'no'
    else:
        param = attack.split('_')[-2:]
        param = [float(p.split(':')[-1]) for p in param]
        steps, eps = param[0], param[1]
        attack_name = ('FGSM' if steps == 1 else 'PGD') + ' ' + str(eps)
    axes[1].set_title('Attention distance under ' + attack_name + ' attack')
    axes[1].legend()
    axes[1].set_aspect(1.0/axes[1].get_data_ratio(), adjustable='box')
    plt.tight_layout()
    plt.savefig('output/val/plots/CKA_AttDist/' + model_1 + '_' + model_2 + attack + '.png')
    plt.close()


def main():
    data = json.load(open('saves/all_summaries_02-22_11:00.json', 'r'))
    attacks = ['_steps:40_eps:0.001', '_steps:40_eps:0.003', '_steps:40_eps:0.005', '_steps:40_eps:0.01',
               '_steps:1_eps:0.031', '_steps:1_eps:0.062']
    list_models = ['t2t_vit_14_p', 't2t_vit_14_t', 't2t_vit_14_t_doexp05l', 't2t_vit_14_t_donegexp05l', 't2t_vit_14_t_donegexp025l',
                   'vit_base_patch16_224_pretrained', 'vit_base_patch32_224_pretrained', 'vit_base_patch32_224_scratch',
                   'vit_base_patch32_224_doexp5', 'vit_base_patch32_224_donegexp025l']
    list_models = ['t2t_vit_14_p', 't2t_vit_14_t', 'vit_base_patch32_224_pretrained', 'vit_base_patch16_224_pretrained']
    get_top1_val(data, attacks, list_models)


if __name__ == '__main__':
    main()
