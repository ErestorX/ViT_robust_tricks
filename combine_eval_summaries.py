from PIL import Image
import numpy as np
import argparse
import csv
import os


parser = argparse.ArgumentParser()
parser.add_argument('--val_path', default='./output/val/', type=str)


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
    models_list = ['_'.join(model.split('_')[:-1]) for model in exp_list]
    models_list = sorted(list(set(models_list)))
    id_exp = ['_'.join(model.split('_')[-1:]) for model in exp_list]
    id_exp = sorted(list(set(id_exp)))
    id_exp.remove('pretrained')
    id_exp.remove('scratch')
    id_exp = ['pretrained', 'scratch'] + id_exp
    folder_list = []
    for model_name in models_list:
        for exp_name in id_exp:
            if os.path.exists(val_path + model_name + '_' + exp_name):
                folder_list.append('_'.join([model_name, exp_name]))
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
    models_list = ['_'.join(model.split('_')[:-1]) for model in exp_list]
    models_list = sorted(list(set(models_list)))
    id_exp = ['_'.join(model.split('_')[-1:]) for model in exp_list]
    id_exp = sorted(list(set(id_exp)))
    id_exp.remove('pretrained')
    id_exp.remove('scratch')
    id_exp = ['pretrained', 'scratch'] + id_exp
    folder_list = []
    for model_name in models_list:
        for exp_name in id_exp:
            if os.path.exists(val_path + model_name + '_' + exp_name):
                folder_list.append('_'.join([model_name, exp_name]))
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


def join_CKA_and_adv_CKA(val_path, exp_list):
    models_list = ['_'.join(model.split('_')[:-1]) for model in exp_list]
    models_list = sorted(list(set(models_list)))
    id_exp = ['_'.join(model.split('_')[-1:]) for model in exp_list]
    id_exp = sorted(list(set(id_exp)))
    id_exp.remove('pretrained')
    id_exp.remove('scratch')
    id_exp = ['pretrained', 'scratch'] + id_exp
    folder_list = []
    for model_name in models_list:
        for exp_name in id_exp:
            if os.path.exists(val_path + model_name + '_' + exp_name):
                folder_list.append('_'.join([model_name, exp_name]))
    for folder_origin in folder_list:
        list_CKA = os.listdir(folder_origin)
        list_CKA = [val_path + file for file in list_CKA if 'CKA' in file]
        list_adv_CKA = [file for file in list_CKA if 'adv' in file]
        list_CKA = [file for file in list_CKA if 'adv' not in file]
        for (clean, adv) in zip(list_CKA, list_adv_CKA):
            im_clean = Image.open(clean).convert('RGB')
            im_adv = Image.open(adv).convert('RGB')
            width = np.asarray(im_clean).shape[1] * 3
            height = np.asarray(im_clean).shape[0]
            new_im = Image.new('RGB', (width, height))
            new_im.paste(im_clean, (0, 0))
            new_im.paste(im_clean - im_adv, (np.asarray(im_clean).shape[1], 0))
            new_im.paste(im_adv, (2 * np.asarray(im_clean).shape[1], 0))
            name_comp = 'CKA_vs_' + adv.split('/')[-1]
            new_im.save(folder_origin + name_comp)


def main():
    args = parser.parse_args()
    exp_list = [exp for exp in os.listdir(args.val_path) if os.path.isdir(args.val_path + exp)]
    summarize_experiments(args.val_path, exp_list)
    for file in os.listdir(args.val_path + exp_list[0]):
        if file.split('.')[-1] == 'png' and file.split('_')[0] != 'CKA':
            summarize_visualization(args.val_path, exp_list, file)
    summarize_CKA(args.val_path, exp_list)
    summarize_adversarial_CKA(args.val_path, exp_list)
    join_CKA_and_adv_CKA(args.val_path, exp_list)


if __name__ == '__main__':
    main()
