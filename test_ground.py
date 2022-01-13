import combine_eval_summaries
import numpy as np
import random
import models
import copy
import json
import matplotlib.pyplot as plt
from models.Custom_T2T import load_custom_t2t_vit
from tabulate import tabulate


def func_1(exp_name):
    data = json.load(open('output/val/all_summaries.json', 'r'))
    list_exp = combine_eval_summaries.order_exp('output/val/', data.keys())
    mat = []
    for id_base, exp_base in enumerate(list_exp):
        line = []
        print(exp_base, len(data[exp_base][exp_name].keys()))
        #     for id_target, exp_target in enumerate(list_exp):
        #         line.append(data[exp_base][exp_name][exp_target])
        #     mat.append(line)
        # mat = np.asarray(mat)


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


def compare_att_distances(model1, model2, adv_ds=False):
    data = json.load(open('saves/all_summaries_01-06_12:00.json', 'r'))
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


def get_top1_val(output_path, data):
    exp_list = data.keys()
    clean_top1 = [data[exp]['Metrics_cln']['top1'] for exp in exp_list]
    adv_top1 = [data[exp]['Metrics_adv']['top1'] for exp in exp_list]
    table = [[exp, cln, adv] for exp, cln, adv in zip(exp_list, clean_top1, adv_top1)]
    print(tabulate(table, headers=['Model', 'Clean top1', 'Adversarial top1']))


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
