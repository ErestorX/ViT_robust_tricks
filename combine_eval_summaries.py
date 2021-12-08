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


def summarize_experiments(val_path, exp_list):
    exp_results = [['Exp', 'Data', 'Loss', 'Acc@1', 'Acc@5']]
    for exp in exp_list:
        with open(val_path + exp + '/Validation.csv', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row_id, row in enumerate(reader):
                if row_id > 0:
                    exp_results.append([exp] + row)
    with open(val_path + 'Experiment_summary.csv', 'w+', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in exp_results:
            writer.writerow(row)


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

def new_summarize_CKAs(val_path, exp_list):
    total_summary = {}
    for exp in exp_list:
        with open(val_path + '/' + exp + '/json_summaries.json') as j_file:
            total_summary[exp] = json.load(j_file)
    fig_CKA = plt.figure()
    fig_adv_CKA = plt.figure()
    fig_diff_CKA = plt.figure()
    for i, exp_i in enumerate(exp_list):
        for j, exp_j in enumerate(exp_list):
            id_ax = 1 + i*len(exp_list) + j
            ax_CKA = fig_CKA.add_subplot(len(exp_list), len(exp_list), id_ax)
            ax_CKA.imshow(total_summary[exp_i][exp_i+'_VS_'+exp_j])
            ax_adv_CKA = fig_adv_CKA.add_subplot(len(exp_list), len(exp_list), id_ax)
            ax_adv_CKA.imshow(total_summary[exp_i]['adv_'+exp_i + '_VS_' + exp_j])
            ax_diff_CKA = fig_diff_CKA.add_subplot(len(exp_list), len(exp_list), id_ax)
            ax_diff_CKA.imshow(total_summary[exp_i][exp_i+'_VS_'+exp_j] - total_summary[exp_i]['adv_'+exp_i + '_VS_' + exp_j])



def test_main():
    args = parser.parse_args()
    exp_list = [exp for exp in os.listdir(args.val_path) if os.path.isdir(args.val_path + exp)]
    ordered_exp_list = order_exp(args.val_path, exp_list)
    full_exp_name = ['custom_' + name if name.split('_')[-1] not in ['pretrained', 'scratch'] else name for name in
                     ordered_exp_list]
    total_summary = {}
    for exp in exp_list:
        with open(args.val_path + exp + '/json_summaries.json') as j_file:
            total_summary[exp] = json.load(j_file)
    clean_top1 = []
    adv_top1 = []
    clean_dist = []
    adv_dist = []
    for exp in ordered_exp_list:
        clean_top1.append(total_summary[exp]['clean_metrics']['top1'])
        adv_top1.append(total_summary[exp]['adv_metrics']['top1'])
        clean_dist.append(np.average(np.asarray(total_summary[exp]['att_distances']), axis=1))
        adv_dist.append(np.average(np.asarray(total_summary[exp]['adv_att_distances']), axis=1))
    clean_dist, adv_dist = np.asarray(clean_dist), np.asarray(adv_dist)
    fig, axes = plt.subplots(1, len(clean_dist[0]), figsize=(48, 4), dpi=200)
    for block_id in range(len(clean_dist[0])):
        for model_id, c, exp in zip(range(len(clean_top1)), cm.rainbow(np.linspace(0, 1, len(clean_top1))), ordered_exp_list):
            axes[block_id].scatter(clean_dist[model_id, block_id], clean_top1[model_id], label=exp, c=c, marker="o")
            axes[block_id].scatter(adv_dist[model_id, block_id], adv_top1[model_id], c=c, marker="D")
        axes[block_id].set_ylim(ymax=100, ymin=0)
        axes[block_id].set_xlim(xmax=160, xmin=60)
    axLine, axLabel = axes[0].get_legend_handles_labels()
    fig.legend(axLine, axLabel, loc='center left')
    plt.show()

    fig, ax = plt.subplots()
    diff_dist = adv_dist - clean_dist
    block_names = [str(i) for i in range(len(diff_dist[0]))]
    for model_id, c, exp in zip(range(len(clean_top1)), cm.rainbow(np.linspace(0, 1, len(clean_top1))), ordered_exp_list):
        ax.scatter(block_names, diff_dist[model_id, :], c=c, label=exp)
    ax.set_ylabel('Difference of attention distance')
    ax.set_xlabel('Block id')
    axLine, axLabel = ax.get_legend_handles_labels()
    fig.legend(axLine, axLabel, loc='lower right')
    plt.show()


def main():
    args = parser.parse_args()
    exp_list = [exp for exp in os.listdir(args.val_path) if os.path.isdir(args.val_path + exp)]
    summarize_experiments(args.val_path, exp_list)
    for file in os.listdir(args.val_path + exp_list[0]):
        if file.split('.')[-1] == 'png' and 'CKA' not in file:
            summarize_visualization(args.val_path, exp_list, file)
    summarize_CKA(args.val_path, exp_list)
    summarize_adversarial_CKA(args.val_path, exp_list)
    summarize_CKA_diff(args.val_path, exp_list)


if __name__ == '__main__':
    main()
    # test_main()
