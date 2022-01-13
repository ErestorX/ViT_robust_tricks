import os.path

import combine_eval_summaries
import numpy as np
import random
import models
import copy
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from models.Custom_T2T import load_custom_t2t_vit
from tabulate import tabulate


def func_1(exp_name):
    data = json.load(open('saves/all_summaries_01-11_17:00.json', 'r'))
    t2t_exp = []
    vit_exp = []
    exp_list = list(data.keys())
    for exp in exp_list:
        if 't2t' in exp:
            t2t_exp.append(exp)
        else:
            vit_exp.append(exp)
    t2t_exp = sorted(t2t_exp)
    exp_list = t2t_exp + vit_exp
    nb_exp = len(exp_list)
    mat = []
    for id_base, exp_base in enumerate(exp_list):
        for id_target, exp_target in enumerate(exp_list):
            mat.append(data[exp_base][exp_name][exp_target])
    fig = plt.figure(figsize=(35, 35))
    for id_i in range(nb_exp ** 2):
        ax_CKA = fig.add_subplot(nb_exp, nb_exp, id_i + 1)
        ax_CKA.imshow(mat[id_i])
        if id_i // nb_exp == 0:
            ax_CKA.set_title(exp_list[int(id_i % nb_exp)])
        if id_i % nb_exp == 0:
            ax_CKA.set_ylabel(exp_list[int(id_i / nb_exp)])
        plt.setp(ax_CKA, xticks=range(0, len(mat[id_i]), 5), yticks=range(0, len(mat[id_i][0]), 5))
    fig.tight_layout()
    plt.savefig('output/val/Summary_' + exp_name + '.png')


def func_2():
    data = json.load(open('saves/all_summaries_01-11_17:00.json', 'r'))
    t2t_exp = []
    vit_exp = []
    exp_list = list(data.keys())
    for exp in exp_list:
        if 't2t' in exp:
            t2t_exp.append(exp)
        else:
            vit_exp.append(exp)
    t2t_exp = sorted(t2t_exp)
    exp_list = t2t_exp + vit_exp
    nb_exp = len(exp_list)
    mat = []
    for id_base, exp_base in enumerate(exp_list):
        mat.append(data[exp_base]['CKA_adv'])
    fig = plt.figure(figsize=(35, 35))
    for id_i in range(nb_exp ** 2):
        ax_CKA = fig.add_subplot(nb_exp, nb_exp, id_i + 1)
        if id_i//nb_exp != id_i%nb_exp:
            plt.setp(ax_CKA, xticks=[], yticks=[])
        else:
            ax_CKA.imshow(mat[id_i//nb_exp])
            plt.setp(ax_CKA, xticks=range(0, len(mat[id_i//nb_exp]), 5), yticks=range(0, len(mat[id_i//nb_exp][0]), 5))
        if id_i // nb_exp == 0:
            ax_CKA.set_title(exp_list[int(id_i % nb_exp)])
        if id_i % nb_exp == 0:
            ax_CKA.set_ylabel(exp_list[int(id_i / nb_exp)])
    fig.tight_layout()
    plt.savefig('output/val/Summary_CKA_adv.png')


def func_3():
    nb_reviewer = 2
    dict_schedule = {}
    list_students = ['Hugo Lemarchant', 'Zongshang Pang', 'Michitaka Yoshida', 'Chenhao Li', 'Bowen Wang', 'Tianran Wu', 'Yewei Song', 'Arashi Fukui', 'Yusuke Hirota', 'Ziyi Chen', 'Anh-Khoa Vo', 'Yankun Wu', 'Tianwei Chen']
    tmp_list = copy.deepcopy(list_students)
    np.random.shuffle(list_students)
    np.random.shuffle(tmp_list)

    # draw the presenter
    for i in list_students:
        dict_schedule[i] = []
        flag_add_i = False
        flag_add_j = False
        # if he is eligible to be reviewer, temporary disable him
        if i in tmp_list:
            tmp_list.remove(i)
            flag_add_i = True
        # draw the reviewer
        for _ in range(nb_reviewer):
            np.random.shuffle(tmp_list)
            # if no reviewer available, refresh the reviewer list
            if len(tmp_list) == 0:
                tmp_list = copy.deepcopy(list_students)
                # disable the presenter
                tmp_list.remove(i)
                flag_add_i = True
                # disable the potentially already selected reviewers
                if len(dict_schedule[i]) > 0:
                    flag_add_j = True
                    j = dict_schedule[i]
                    for _j in j:
                        tmp_list.remove(_j)
                np.random.shuffle(tmp_list)
            dict_schedule[i].append(tmp_list.pop(0))
        if flag_add_i:
            tmp_list.append(i)
        if flag_add_j:
            for _j in j:
                tmp_list.append(_j)

    print(dict_schedule)


def compare_att_distances_2(model1, model2, adv_ds=False):
    if os.path.exists('output/val/AttDist_' + ('adv_' if adv_ds else 'cln_') + model2 + 'VS' + model1 + '.png'):
        return
    data = json.load(open('saves/all_summaries_01-11_14:15.json', 'r'))
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
    plt.axvline(x=-.5, color='grey', alpha=0.5)
    plt.savefig('output/val/AttDist_' + ('adv_' if adv_ds else 'cln_') + model1 + 'VS' + model2 + '.png')
    plt.close()


def compare_att_distances(model1):
    data = json.load(open('saves/all_summaries_01-11_14:15.json', 'r'))
    exp1 = data[model1]['AttDist_cln']
    exp2 = data[model1]['AttDist_adv']
    if 't2t' in model1:
        blocks1 = range(-2, len(exp1) - 2)
        blocks2 = np.arange(-1.7, len(exp2) - 1.7, 1.0)
    else:
        blocks1 = range(len(exp1))
        blocks2 = np.arange(.3, len(exp2) + .3, 1.0)
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
    ax.set_xticks(blocks1)
    plt.axvline(x=-.5, color='grey', alpha=0.5)
    plt.savefig('output/val/AttDist_' + model1 + '.png')
    plt.close()


def get_top1_val(output_path, data):
    exp_list = data.keys()
    clean_top1 = [data[exp]['Metrics_cln']['top1'] for exp in exp_list]
    adv_top1 = [data[exp]['Metrics_adv']['top1'] for exp in exp_list]
    table = [[exp, cln, adv] for exp, cln, adv in zip(exp_list, clean_top1, adv_top1)]
    print(tabulate(table, headers=['Model', 'Clean top1', 'Adversarial top1']))


def plot_cleanacc_vs_advacc(data):
    exp_list = list(data.keys())
    vit_list = [exp for exp in exp_list if 't2t' not in exp]
    t2t_list = [exp for exp in exp_list if 't2t' in exp]
    exp_list = t2t_list + vit_list
    clean_acc = [data[exp]['Metrics_cln']['top1'] for exp in exp_list]
    adv_acc = [data[exp]['Metrics_adv']['top1'] for exp in exp_list]
    fig, ax = plt.subplots()
    plt.title('Clean accuracy vs. adversarial accuracy')
    colors = cm.rainbow(np.linspace(0, 1, len(exp_list)))

    for c, clean, adv, exp in zip(colors, clean_acc, adv_acc, exp_list):
        if 'do' in exp:
            ax.scatter(clean, adv, marker='s', color=c, label=exp)
        elif 'scratch' in exp:
            ax.scatter(clean, adv, marker='^', color=c, label=exp)
        else:
            ax.scatter(clean, adv, marker='o', color=c, label=exp)
    ax.set_xlabel('Clean accuracy')
    ax.set_ylabel('Adversarial accuracy')
    ax.legend(fontsize=6)
    plt.savefig('output/val/Clean_vs_Adversarial_acc.png')


def get_CKA_adv_plot(output_path, data):
    exp_list = data.keys()
    CKA_adv_list = [data[exp]['CKA_adv'] for exp in exp_list if 'CKA_adv' in data[exp].keys()]
    for exp, CKA_adv in zip(exp_list, CKA_adv_list):
        fig = plt.figure(figsize=(8, 8))
        fig.suptitle(exp)
        ax_CKA = fig.add_subplot(111)
        CKA_adv = np.asarray(CKA_adv)
        ax_CKA.imshow(CKA_adv)
        fig.tight_layout()
        plt.savefig(output_path + 'CKA_adv_' + exp + '.png')

if __name__ == '__main__':
    # get_top1_val('output/val/', json.load(open('saves/all_summaries_01-11_14:15.json', 'r')))
    # get_CKA_adv_plot('output/val/', json.load(open('saves/all_summaries_01-11_14:15.json', 'r')))
    data = json.load(open('saves/all_summaries_01-11_17:00.json', 'r'))
    exp = list(data.keys())
    for e1 in exp:
        compare_att_distances(e1)
        for e2 in exp:
            compare_att_distances_2(e1, e2, adv_ds=False)
            compare_att_distances_2(e1, e2, adv_ds=True)
    # func_1('CKA_cln')
    # func_2()
    # plot_cleanacc_vs_advacc(json.load(open('saves/all_summaries_01-11_17:00.json', 'r')))
