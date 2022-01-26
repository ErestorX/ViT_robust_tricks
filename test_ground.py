import numpy as np

from combine_eval_summaries import *


def get_top1_val(data, experiments, model_list):
    t2t_blue = ['midnightblue', 'mediumblue', 'blue', 'mediumslateblue', 'darkorchid', 'fuchsia', 'violet', 'hotpink',
                'pink']
    t2t_blue.reverse()
    blue_id = 0
    vit_green = ['darkgreen', 'green', 'limegreen', 'mediumseagreen', 'aqquamarine', 'turquoise', 'paleturquoise',
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
        plt.plot(exp_title, per_model_evol[i], label=model_list[i], color=color)
    plt.axvline(x=exp_title[1], color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=exp_title[5], color='gray', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('output/val/Clean_vs_Adversarial_acc.png')


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
            # ax.set_xticklabels(range(-6 if i < 4 else 0, len(), 3))
            # ax.set_yticklabels(range(-6 if j < 4 else 0, 3))
    fig.tight_layout()
    plt.savefig('output/val/plots/' + cka_type + '.png')

def plot_cka_adv(data, cka_type):
    list_models = ['t2t_vit_14_p', 't2t_vit_14_t', 't2t_vit_14_t_doexp05l', 't2t_vit_14_t_donegexp05l',
                   'vit_base_patch16_224_pretrained', 'vit_base_patch32_224_pretrained', 'vit_base_patch32_224_scratch',
                   'vit_base_patch32_224_doexp5']
    mat = []
    for model_b in list_models:
        mat.append(data[model_b][cka_type])
    fig = plt.figure(figsize=(35, 35))
    for i in range(len(list_models)):
        for j in range(len(list_models)):
            if i != j:
                continue
            ax = fig.add_subplot(len(list_models), len(list_models), i * len(list_models) + j + 1)
            ax.imshow(mat[i])
            ax.set_xlabel(list_models[j] + ' clean data')
            ax.set_ylabel('adversarial data')
            # ax.set_xticklabels(range(-6 if i < 4 else 0, len(), 3))
            # ax.set_yticklabels(range(-6 if j < 4 else 0, 3))
    fig.tight_layout()
    plt.savefig('output/val/plots/' + cka_type + '.png')


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
                # print('    {}'.format(dicb[key]))
                # print('    {}'.format(dica[key]))
                # dica[key] = dicb[key]
        else:
            dica[key] = dicb[key]
    return dica


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
                legends.append(model)
            bp = ax.boxplot(distance_points, positions=acc, vert=False, patch_artist=True, widths=1.5)
            for patch, color in zip(bp['boxes'], t2t_blue[:blue_id]):
                patch.set(facecolor=color)
        else:
            for model in data_figs[block].keys():
                acc.append(round(data_figs[block][model][0], 2))
                distance_points.append(data_figs[block][model][1])
                legends.append(model)
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


if __name__ == '__main__':
    json_file = 'output/val/all_summaries.json'
    data = json.load(open(json_file, 'r'))
    attacks = ['_steps:40_eps:0.001', '_steps:40_eps:0.003', '_steps:40_eps:0.005', '_steps:40_eps:0.01', '_steps:1_eps:0.031', '_steps:1_eps:0.062']
    list_models = ['t2t_vit_14_p', 't2t_vit_14_t', 't2t_vit_14_t_doexp05l', 't2t_vit_14_t_donegexp05l', 'vit_base_patch16_224_pretrained', 'vit_base_patch32_224_pretrained', 'vit_base_patch32_224_scratch', 'vit_base_patch32_224_doexp5']

    # plot_cka_mat(data, 'CKA_single_cln')
    # plot_cka_adv(data, 'CKA_single_adv_steps:40_eps:0.01')
    # for m1 in list_models:
    #     compare_att_distances_model(data, m1, attacks)
    # compare_att_distances_attack(data, 'AttDist_cln', list_models)
    # for a1 in attacks:
    #     a1 = 'AttDist_adv' + a1
    #     compare_att_distances_attack(data, a1, list_models)
    # dica = json.load(open('saves/all_summaries_01-17_13:00.json', 'r'))
    # dicb = data
    # dica = recursive_merge_dictionaries(dica, dicb)
    # json.dump(dica, open('output/val/all_summaries.json', 'w'))
    # get_top1_val(json.load(open(json_file, 'r')), attacks, list_models)
    # AttDist_vs_top1(data, '_cln', list_models)
    # for a1 in attacks:
    #     AttDist_vs_top1(data, '_adv'+a1, list_models)
    for model in data:
        experiments = list(data[model].keys())
        for exp in experiments:
            if 'AttDist' in exp:
                data[model].pop(exp)
    json.dump(data, open(json_file, 'w'))
