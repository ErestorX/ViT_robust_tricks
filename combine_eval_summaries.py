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


def compare_att_distances(data, model1):
    exp1 = data[model1]['AttDist_cln']
    exp2 = data[model1]['AttDist_adv']
    if 't2t' in model1:
        blocks1 = range(-2, len(exp1)-2)
        blocks2 = np.arange(-1.7, len(exp2)-1.7, 1.0)
    else:
        blocks1 = range(len(exp1))
        blocks2 = np.arange(.3, len(exp2)+.3, 1.0)
    fig, ax = plt.subplots()
    plt.title('Attention distance on ' + model1)
    bp1 = ax.boxplot(exp1, positions=blocks1, patch_artist=True, widths=0.3)
    bp2 = ax.boxplot(exp2, positions=blocks2, patch_artist=True, widths=0.3)
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp1[element], color='black')
        plt.setp(bp2[element], color='r')
    for patch in bp1['boxes']:
        patch.set(facecolor='None')
    for patch in bp2['boxes']:
        patch.set(facecolor='red', alpha=0.5)
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Clean data', 'Adversarial data'])
    ax.set_ylim(0, 224)
    ax.yaxis.grid(True)
    blocks1 = list(blocks1)
    blocks2 = np.round(blocks2-.3, 0).astype(np.int).tolist()
    blocks = sorted(list(set(blocks1 + blocks2)))
    ax.set_xticks(blocks)
    if 't2t' in model1:
        plt.axvline(x=-.5, color='grey', alpha=0.5)
    plt.savefig('output/val/AttDist_' + model1 + '.png')


def compare_att_distances_2(data, model1, model2, adv_ds=False):
    exp1 = data[model1]['AttDist_adv' if adv_ds else 'AttDist_cln']
    exp2 = data[model2]['AttDist_adv' if adv_ds else 'AttDist_cln']
    if 't2t' in model1:
        blocks1 = range(-2, len(exp1)-2)
    else:
        blocks1 = range(len(exp1))
    if 't2t' in model2:
        blocks2 = np.arange(-1.7, len(exp2)-1.7, 1.0)
    else:
        blocks2 = np.arange(.3, len(exp2)+.3, 1.0)
    fig, ax = plt.subplots()
    plt.title('Attention distance on ' + ('adversarial data' if adv_ds else 'clean data'))
    bp1 = ax.boxplot(exp1, positions=blocks1, patch_artist=True, widths=0.3)
    bp2 = ax.boxplot(exp2, positions=blocks2, patch_artist=True, widths=0.3)
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp1[element], color='black')
        plt.setp(bp2[element], color='r')
    for patch in bp1['boxes']:
        patch.set(facecolor='None')
    for patch in bp2['boxes']:
        patch.set(facecolor='red', alpha=0.5)
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], [model1, model2])
    ax.set_ylim(0, 224)
    ax.yaxis.grid(True)
    blocks1 = list(blocks1)
    blocks2 = np.round(blocks2-.3, 0).astype(np.int).tolist()
    blocks = sorted(list(set(blocks1 + blocks2)))
    ax.set_xticks(blocks)
    if 't2t' in model1 or 't2t' in model2:
        plt.axvline(x=-.5, color='grey', alpha=0.5)
    plt.savefig('output/val/AttDist_' + ('adv_' if adv_ds else 'cln_') + model1 + 'VS' + model2 + '.png')


def get_top1_val(data):
    exp_list = data.keys()
    clean_top1 = [data[exp]['Metrics_cln']['top1'] for exp in exp_list]
    adv_top1 = [data[exp]['Metrics_adv']['top1'] for exp in exp_list]
    table = [[exp, cln, adv] for exp, cln, adv in zip(exp_list, clean_top1, adv_top1)]
    print(tabulate(table, headers=['Model', 'Clean top1', 'Adversarial top1']))


def main():
    data = json.load(open('saves/all_summaries_01-11_17:00.json', 'r'))
    get_top1_val(data)
    exp = list(data.keys())
    for e1 in exp:
        compare_att_distances(data, e1)
        for e2 in exp:
            compare_att_distances_2(data, e1, e2, adv_ds=False)
            compare_att_distances_2(data, e1, e2, adv_ds=True)


if __name__ == '__main__':
    main()
