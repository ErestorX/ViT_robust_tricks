import combine_eval_summaries
import numpy as np
import random
import models
import copy
import json


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


def func2():
    list_of_individual = ['Hugo', 'Linh', 'Koki', 'Tomomi', 'Aria', 'Sy', 'Ferin', 'Rizaldi', 'Luca', 'Tianwei', 'Fan', 'Fan_s fiend',
            'Ozgur', 'Jimmy']
    pairs = []
    copy_list = list_of_individual.copy()
    for id in range(len(list_of_individual)):
        rand_id = id
        while rand_id == id:
            rand_id = random.randint(0, len(copy_list) - 1)
            if list_of_individual[id] != copy_list[rand_id]:
                pairs.append([list_of_individual[id], copy_list[rand_id]])
                copy_list.pop(rand_id)
    return pairs


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


if __name__ == '__main__':
    func_3()
