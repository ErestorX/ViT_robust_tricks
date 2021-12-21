import json

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import numpy as np
import argparse
import csv
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


def summarize_CKA(val_path, exp_list):
    folder_list = order_exp(val_path, exp_list)
    full_exp_name = ['custom_' + name if name.split('_')[-1] not in ['pretrained', 'scratch'] else name for name in folder_list]
    new_im, total_width, total_height = None, None, None
    for col_id, (folder_origin, exp_origin) in enumerate(zip(folder_list, full_exp_name)):
        for row_id, exp_target in enumerate(full_exp_name):
            if row_id < col_id:
                continue
            if exp_origin == exp_target:
                file_name = val_path + folder_origin + '/CKA_' + exp_origin + '.png'
            else:
                file_name = val_path + folder_origin + '/CKA_' + exp_origin + '_|_' + exp_target + '.png'
            im = Image.open(file_name).convert('RGB')
            if new_im is None:
                total_width = np.asarray(im).shape[1] * len(folder_list)
                total_height = np.asarray(im).shape[0] * len(folder_list)
                new_im = Image.new('RGB', (total_width, total_height))
                new_im.paste(im, (col_id * np.asarray(im).shape[1], row_id * np.asarray(im).shape[0]))
            new_im.paste(im, (col_id * np.asarray(im).shape[1], row_id * np.asarray(im).shape[0]))
    new_im.save(val_path + 'Summary_CKA.png')


def summarize_adversarial_CKA(val_path, exp_list):
    folder_list = order_exp(val_path, exp_list)
    full_exp_name = ['custom_' + name if name.split('_')[-1] not in ['pretrained', 'scratch'] else name for name in folder_list]
    new_im, total_width, total_height = None, None, None
    for col_id, (folder_origin, exp_origin) in enumerate(zip(folder_list, full_exp_name)):
        for row_id, exp_target in enumerate(full_exp_name):
            if exp_origin == exp_target:
                file_name = val_path + folder_origin + '/CKA_adv_' + exp_origin + '.png'
            else:
                file_name = val_path + folder_origin + '/CKA_adv_' + exp_origin + '_|_' + exp_target + '.png'
            im = Image.open(file_name).convert('RGB')
            if new_im is None:
                total_width = np.asarray(im).shape[1] * len(folder_list)
                total_height = np.asarray(im).shape[0] * len(folder_list)
                new_im = Image.new('RGB', (total_width, total_height))
                new_im.paste(im, (col_id * np.asarray(im).shape[1], row_id * np.asarray(im).shape[0]))
            new_im.paste(im, (col_id * np.asarray(im).shape[1], row_id * np.asarray(im).shape[0]))
    new_im.save(val_path + 'Summary_adversarial_CKA.png')


def summarize_CKA_diff(val_path, exp_list):
    folder_list = order_exp(val_path, exp_list)
    full_exp_name = ['custom_' + name if name.split('_')[-1] not in ['pretrained', 'scratch'] else name for name in folder_list]
    new_im, total_width, total_height = None, None, None
    for col_id, (folder_origin, exp_origin) in enumerate(zip(folder_list, full_exp_name)):
        for row_id, exp_target in enumerate(full_exp_name):
            file_name = val_path + folder_origin + '/diff_CKA_' + exp_origin + '_|_' + exp_target + '.png'
            im = Image.open(file_name).convert('RGB')
            if new_im is None:
                total_width = np.asarray(im).shape[1] * len(folder_list)
                total_height = np.asarray(im).shape[0] * len(folder_list)
                new_im = Image.new('RGB', (total_width, total_height))
                new_im.paste(im, (col_id * np.asarray(im).shape[1], row_id * np.asarray(im).shape[0]))
            new_im.paste(im, (col_id * np.asarray(im).shape[1], row_id * np.asarray(im).shape[0]))
    new_im.save(val_path + 'Summary_CKA_diff.png')


def new_summarize_CKAs(val_path, data):
    exp_list = list(data.keys())
    final_exp_list = exp_list.copy()
    for exp_origin in exp_list:
        for exp_target in exp_list:
            if exp_target not in data[exp_origin]['CKA']:
                final_exp_list.remove(exp_target)


    # fig_CKA = plt.figure()
    # fig_adv_CKA = plt.figure()
    # fig_diff_CKA = plt.figure()
    # for i, exp_i in enumerate(exp_list):
    #     for j, exp_j in enumerate(exp_list):
    #         id_ax = 1 + i*len(exp_list) + j
    #         ax_CKA = fig_CKA.add_subplot(len(exp_list), len(exp_list), id_ax)
    #         ax_CKA.imshow(total_summary[exp_i][exp_i+'_VS_'+exp_j])


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
    json.dump(total_summary, open(val_path + '/json_summaries_combined.json', 'w+'))
    return total_summary


def main():
    args = parser.parse_args()
    exp_list = [exp for exp in os.listdir(args.val_path) if os.path.isdir(args.val_path + exp)]
    exp_list = order_exp(args.val_path, exp_list)
    data = combine_jsons(args.val_path, exp_list)
    # exp_list = list(data.keys())
    # for file in os.listdir(args.val_path + exp_list[0]):
    #     if file.split('.')[-1] == 'png' and 'CKA' not in file:
    #         summarize_visualization(args.val_path, exp_list, file)
    new_summarize_CKAs(args.val_path, data)
    # summarize_CKA(args.val_path, exp_list)
    # summarize_adversarial_CKA(args.val_path, exp_list)
    # summarize_CKA_diff(args.val_path, exp_list)


if __name__ == '__main__':
    main()
