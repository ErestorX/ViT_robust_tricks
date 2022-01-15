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
