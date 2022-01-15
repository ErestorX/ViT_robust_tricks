import combine_eval_summaries
import numpy as np
import random
import models
import copy
import json
import matplotlib.pyplot as plt
from models.Custom_T2T import load_custom_t2t_vit
from tabulate import tabulate


def plot_cka_mat(data, cka_type):
    list_models = ['t2t_vit_14_p', 't2t_vit_14_t', 't2t_vit_14_t_doexp05l', 't2t_vit_14_t_donegexp05l', 'vit_base_patch16_224_pretrained', 'vit_base_patch32_224_pretrained', 'vit_base_patch32_224_scratch', 'vit_base_patch32_224_doexp5']
    mat = []
    for model_b in list_models:
        for model_t in list_models:
            mat.append(data[model_b][cka_type][model_t])
    fig = plt.figure(figsize=(35, 35))
    for i in range(len(list_models)):
        for j in range(len(list_models)):
            ax = fig.add_subplot(len(list_models), len(list_models), i*len(list_models)+j+1)
            ax.imshow(mat[i*len(list_models)+j])
            if i == 0:
                ax.set_title(list_models[j])
            if j == 0:
                ax.set_ylabel(list_models[i])
            ax.set_xticklabels(range(-6 if i < 4 else 0, step=3))
            ax.set_yticklabels(range(-6 if j < 4 else 0, step=3))
    fig.tight_layout()
    plt.savefig(cka_type + '.png')


def summarize_dists(val_path, data, adv_ds=False):
    key_dist = 'AttDist_adv' if adv_ds else 'AttDist_cln'
    exp_list = ['t2t_vit_14_p', 't2t_vit_14_t', 'vit_base_patch16_224_pretrained', 'vit_base_patch32_224_pretrained',
                'vit_base_patch32_224_scratch', 'vit_small_patch16_224_pretrained', 'vit_small_patch32_224_pretrained',
                'vit_tiny_patch16_224_pretrained', 'vit_tiny_patch16_224_scratch']
    exp_list = [exp for exp in exp_list if exp in data.keys() and key_dist in data[exp].keys()]
    values = []
    avg_values = []
    for exp in exp_list:
        values.append(np.asarray(data[exp][key_dist]))
    for exp, value in zip(exp_list, values):
        plt.title(key_dist + ' ' + exp)
        plt.boxplot(value.swapaxes(0, 1), widths=0.5)
        avg_values.append(np.mean(value, axis=1))
        plt.ylim(0, 224)
        plt.savefig(val_path + exp + '_' + key_dist + '.png')
        plt.clf()
    for exp, value in zip(exp_list, avg_values):
        nb_blocks = value.shape[0]
        x = np.arange(1, nb_blocks+1)
        plt.scatter(x, value, label=exp)
    plt.title('All experiments ' + key_dist)
    plt.legend()
    plt.ylim(0, 224)
    plt.savefig(val_path + 'all_' + key_dist + '.png')
    plt.clf()


def get_CKA_adv_plot(output_path, data):
    exp_list = data.keys()
    CKA_adv_list = [data[exp]['CKA_adv'] for exp in exp_list if 'CKA_adv' in data[exp].keys()]
    print(len(CKA_adv_list))
    for exp, CKA_adv in zip(exp_list, CKA_adv_list):
        fig = plt.figure(figsize=(8, 8))
        fig.suptitle(exp)
        ax_CKA = fig.add_subplot(111)
        CKA_adv = np.asarray(CKA_adv)
        ax_CKA.imshow(CKA_adv)
        fig.tight_layout()
        plt.savefig(output_path + exp + '_CKA_adv.png')


def recursive_merge_dictionaries(dica, dicb, tree=None):
    if tree is None:
        tree=[]
    for key in dicb:
        if key in dica:
            if isinstance(dicb[key], dict) and isinstance(dica[key], dict):
                next_tree = tree + [key]
                recursive_merge_dictionaries(dica[key], dicb[key], next_tree)
            elif dicb[key] == dica[key]:
                pass
            else:
                print('Conflict: {}'.format('.'.join(tree + [key])))
                print('    {}'.format(dicb[key]))
                print('    {}'.format(dica[key]))
                print('Usinsg value from second dict')
                dica[key] = dicb[key]
        else:
            dica[key] = dicb[key]
    return dica


if __name__ == '__main__':
    # get_top1_val('output/val/', json.load(open('saves/all_summaries_01-11_14:15.json', 'r')))
    # get_CKA_adv_plot('output/val/', json.load(open('saves/all_summaries_01-11_14:15.json', 'r')))
    # func_1('CKA_cln')
    # func_2()
    # plot_cleanacc_vs_advacc(json.load(open('saves/all_summaries_01-11_17:00.json', 'r')))
    # data = json.load(open('output/val/all_summaries.json', 'r'))
    # for exp in data.keys():
    #     metrics_adv = data[exp].pop('Metrics_adv')
    #     CKA_adv = data[exp].pop('CKA_adv')
    #     CKA_trf = data[exp].pop('CKA_trf')
    #     data[exp]['Metrics_adv_steps:1_eps:0.0062'] = metrics_adv
    #     data[exp]['CKA_adv_steps:1_eps:0.0062'] = CKA_adv
    #     data[exp]['CKA_trf_steps:1_eps:0.0062'] = CKA_trf
    # json.dump(data, open('output/val/all_summaries.json', 'w'))
    # dica = json.load(open('output/val/all_summaries.json', 'r'))
    # dicb = json.load(open('output/val/all_summaries_2.json', 'r'))
    # dica = recursive_merge_dictionaries(dica, dicb)
    # json.dump(dica, open('output/val/all_summaries.json', 'w'))
    pass
