import json

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import numpy as np
import argparse
import csv
import os

import models.T2T

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


def summarize_visualization(val_path, exp_list, file):
    models_list = ['_'.join(model.split('_')[:4]) for model in exp_list]
    models_list = sorted(list(set(models_list)))
    id_exp = ['_'.join(model.split('_')[4:]) for model in exp_list]
    id_exp = sorted(list(set(id_exp)))
    id_exp.remove('pretrained')
    id_exp.remove('scratch')
    id_exp = ['pretrained', 'scratch'] + id_exp
    new_im, total_width, total_height = None, None, None
    for row_id, model in enumerate(models_list):
        models_exp = [exp for exp in exp_list if model in exp]
        for exp in models_exp:
            col_id = id_exp.index(exp.split('_')[4])
            im = Image.open(val_path + exp + '/' + file).convert('RGB')
            if new_im is None:
                total_width = np.asarray(im).shape[1] * len(id_exp)
                total_height = np.asarray(im).shape[0] * len(models_list)
                new_im = Image.new('RGB', (total_width, total_height))
                new_im.paste(im, (col_id * np.asarray(im).shape[1], row_id * np.asarray(im).shape[0]))
            new_im.paste(im, (col_id * np.asarray(im).shape[1], row_id * np.asarray(im).shape[0]))
    new_im.save(val_path + 'Summary_' + file)


def summarize_dists(val_path, data):

    pass


def summarize_CKAs(val_path, data, exp_code='CKA_cln'):
    # exp_list = list(data.keys())
    exp_list = ['t2t_vit_14_p', 't2t_vit_14_t', 'vit_base_patch16_224_pretrained', 'vit_base_patch32_224_pretrained',
                'vit_base_patch32_224_scratch', 'vit_small_patch16_224_pretrained', 'vit_small_patch32_224_pretrained',
                'vit_tiny_patch16_224_pretrained', 'vit_tiny_patch16_224_scratch']
    final_exp_list = exp_list.copy()
    nb_exp = len(final_exp_list)
    arg_min, arg_max = 1e99, -1e99
    mat_results = []
    for id_i, i in enumerate(final_exp_list):
        for id_j, j in enumerate(final_exp_list):
            if j not in data[i][exp_code]:
                tmp_mat = np.zeros((1, 1)) * 0.5
            else:
                tmp_mat = np.asarray(data[i][exp_code][j])
            mat_results.append(tmp_mat)
            arg_min = np.amin(tmp_mat) if np.amin(tmp_mat) < arg_min else arg_min
            arg_max = np.amax(tmp_mat) if np.amax(tmp_mat) > arg_max else arg_max
    for id_i in range(nb_exp ** 2):
        mat_results[id_i] = (mat_results[id_i] - arg_min) / (arg_max - arg_min)
    fig = plt.figure(figsize=(35, 35))
    for id_i in range(nb_exp ** 2):
        ax_CKA = fig.add_subplot(nb_exp, nb_exp, id_i + 1)
        ax_CKA.imshow(mat_results[id_i])
        if id_i // nb_exp == 0:
            ax_CKA.set_title(final_exp_list[int(id_i % nb_exp)])
        if id_i % nb_exp == 0:
            ax_CKA.set_ylabel(final_exp_list[int(id_i / nb_exp)])
        plt.setp(ax_CKA, xticks=range(0, len(mat_results[id_i]), 5), yticks=range(0, len(mat_results[id_i][0]), 5))
    fig.tight_layout()
    plt.savefig(val_path + 'Summary_' + exp_code + '.png')
    return mat_results, final_exp_list


def combine_jsons(val_path, exp_list):
    total_summary = {}
    for exp in exp_list:
        with open(val_path + '/' + exp + '/json_summaries.json') as j_file:
            total_summary[exp] = json.load(j_file)
    for summary in total_summary:
        total_summary[summary]['AttDist_cln'] = total_summary[summary].pop('att_distances')
        total_summary[summary]['AttDist_adv'] = total_summary[summary].pop('adv_att_distances')
        total_summary[summary]['Metrics_cln'] = total_summary[summary].pop('clean_metrics')
        total_summary[summary]['Metrics_adv'] = total_summary[summary].pop('adv_metrics')
        total_summary[summary]['CKA_cln'] = {}
        total_summary[summary]['CKA_adv'] = {}
        keys = list(total_summary[summary].keys())
        for key in keys:
            if summary in key:
                CKA_name = ''.join(key.split('_VS_'))
                CKA_name = CKA_name.split(summary)
                for id, val in enumerate(CKA_name):
                    if 'custom_' in val:
                        CKA_name[id] = ''.join(val.split('custom_'))
                if CKA_name[0] == 'adv_':
                    if CKA_name[1] == '':
                        total_summary[summary]['CKA_adv'][summary] = total_summary[summary].pop(key)
                    else:
                        total_summary[summary]['CKA_adv'][CKA_name[1]] = total_summary[summary].pop(key)
                else:
                    if CKA_name[1] == '':
                        total_summary[summary]['CKA_cln'][summary] = total_summary[summary].pop(key)
                    else:
                        total_summary[summary]['CKA_cln'][CKA_name[1]] = total_summary[summary].pop(key)
    json.dump(total_summary, open(val_path + '/all_summaries.json', 'w+'))
    return total_summary


def main():
    args = parser.parse_args()
    exp_list = [exp for exp in os.listdir(args.val_path) if os.path.isdir(args.val_path + exp)]
    exp_list = order_exp(args.val_path, exp_list)
    data = combine_jsons(args.val_path, exp_list)
    result_cln, final_exp_list = summarize_CKAs(args.val_path, data, 'CKA_cln')
    result_adv, _ = summarize_CKAs(args.val_path, data, 'CKA_adv')
    nb_exp = len(final_exp_list)
    fig = plt.figure(figsize=(35, 35))
    mat_results = []
    for adv, cln in zip(result_adv, result_cln):
        mat_results.append(adv - cln)
    for id_i in range(nb_exp ** 2):
        ax_CKA = fig.add_subplot(nb_exp, nb_exp, id_i + 1)
        ax_CKA.imshow(mat_results[id_i])
        if id_i // nb_exp == 0:
            ax_CKA.set_title(final_exp_list[int(id_i % nb_exp)])
        if id_i % nb_exp == 0:
            ax_CKA.set_ylabel(final_exp_list[int(id_i / nb_exp)])
        plt.setp(ax_CKA, xticks=range(0, len(mat_results[id_i]), 5), yticks=range(0, len(mat_results[id_i][0]), 5))
    fig.tight_layout()
    plt.savefig(args.val_path + 'Summary_CKA_dif.png')
    # summarize_CKA_diff(args.val_path, exp_list)


if __name__ == '__main__':
    main()
