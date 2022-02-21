import random
from utils_eval import *
import torch
import models
from torch.nn import CrossEntropyLoss
from copy import deepcopy


def get_all_hooks_non_distributed(model, is_t2t=False, is_performer=False):
    hooks = []
    if is_t2t:
        if is_performer:
            list_targets = ['tokens_to_token.attention1.kqv', 'tokens_to_token.attention1.proj',
                            'tokens_to_token.attention1.mlp', 'tokens_to_token.attention2.kqv',
                            'tokens_to_token.attention2.proj', 'tokens_to_token.attention2.mlp']
        else:
            list_targets = ['tokens_to_token.attention1.attn.qkv', 'tokens_to_token.attention1.attn.proj',
                            'tokens_to_token.attention1.mlp', 'tokens_to_token.attention2.attn.qkv',
                            'tokens_to_token.attention2.attn.proj', 'tokens_to_token.attention2.mlp']
        for tgt in list_targets:
            hook = HookedCache(model, tgt)
            hooks.append(hook)
        hook = HookedCache(model, 'tokens_to_token.project')
        hooks.append(hook)
    for j, block in enumerate(model.blocks):
        hook = HookedCache(model, 'blocks.{}.attn.qkv'.format(j))
        hooks.append(hook)
        hook = HookedCache(model, 'blocks.{}.attn.proj'.format(j))
        hooks.append(hook)
        hook = HookedCache(model, 'blocks.{}.mlp'.format(j))
        hooks.append(hook)
    return hooks


def prediction_step(model_c, model_c_name, input, target, loss_fn, attack):
    pgd_steps, step_size, epsilonMax = attack
    model_t = deepcopy(model_c)
    model_t_name = model_c_name
    writer = SummaryWriter()
    modc_hooks = get_all_hooks_non_distributed(model_c, is_t2t='t2t' in model_c_name,
                                               is_performer=model_c_name.split('_')[3] == 'p')
    modt_hooks = get_all_hooks_non_distributed(model_t, is_t2t='t2t' in model_t_name,
                                               is_performer=model_t_name.split('_')[3] == 'p')
    metrics_ct = make_pairwise_metrics(modc_hooks, modt_hooks)

    input_orig = input.clone()
    for _ in range(pgd_steps):
        input.requires_grad = True
        output = model_c(input)
        model_c.zero_grad()
        cost = loss_fn(output, target)
        grad = torch.autograd.grad(cost, input, retain_graph=False, create_graph=False)[0]
        input = input + step_size * grad.sign()
        input = input_orig + torch.clamp(input - input_orig, -epsilonMax, epsilonMax)
        input = torch.clamp(input, -1, 1).detach()
    _ = model_c(input)
    output = model_t(input)
    update_metrics(modc_hooks, modt_hooks, metrics_ct, "cka/ct", 0, writer, True)
    for hook0 in modc_hooks:
        for hook1 in modt_hooks:
            hook0.clear()
            hook1.clear()
    sim_mat = get_simmat_from_metrics(metrics_ct)
    return torch.argmax(output, axis=1), sim_mat


def print_simmat_var(sim_1, sim_2, out_1, out_2, tar, val_path, im_id, model_name, atk_1, atk_2):
    diagonal_sim_1, diagonal_sim_2 = [], []
    non_diagonal_sim_1, non_diagonal_sim_2 = [], []
    for i in range(len(sim_1)):
        diagonal_sim_1.append(sim_1[i, i])
        diagonal_sim_2.append(sim_2[i, i])
        for j in range(len(sim_1)):
            if i != j:
                non_diagonal_sim_1.append(sim_1[i, j])
                non_diagonal_sim_2.append(sim_2[i, j])
    plt.title('\n'.join([model_name, 'Real: ' + str(tar) + ' Original pred: ' + str(out_1) + ' Detect pred' + str(
                             out_2)]))
    c = 'r'
    bp_cln = plt.boxplot([diagonal_sim_1, non_diagonal_sim_1], positions=[0, 1.5], patch_artist=True,
                         boxprops=dict(facecolor=c, color='black', alpha=0.5), medianprops=dict(color='black'))
    c = 'g'
    bp_trf = plt.boxplot([diagonal_sim_2, non_diagonal_sim_2], positions=[0.33, 1.83], patch_artist=True,
                         boxprops=dict(facecolor=c, color='black', alpha=0.5), medianprops=dict(color='black'))
    plt.xticks([0, 1.5], ['Diagonal values', 'Non-diagonal values'])
    plt.ylim(0, 1.1)
    for val in [.2, .4, .6, .8, 1]:
        plt.axhline(y=val, alpha=0.5, color='black')
    plt.legend([bp_cln['boxes'][0], bp_trf['boxes'][0]], [atk_1, atk_1+' then '+atk_2])
    plt.ylabel('CKA variations')
    if np.array_equal(out_1, tar):
        plt.savefig(val_path + '/correct_preds/class_' + str(im_id) + '.png')
    else:
        plt.savefig(val_path + '/incorrect_preds/class_' + str(im_id) + '.png')
    plt.close()


def main(adv_atk=0, detect_atk=1, version=2, ckpt='t', p=True, data='/media/hugo/Data/ImageNet'):
    torch.cuda.set_device(0)
    attacks = [[0, 0, 0], [1, 1, 0.062], [1, 1, 0.031], [40, 0.025, 0.001], [40, 0.025, 0.003], [40, 0.025, 0.005],
               [40, 0.025, 0.01]]
    attacks_name = ['clean', 'FGSM-0.062', 'FGSM-0.031', 'PGD-0.001', 'PGD-0.003', 'PGD-0.005', 'PGD-0.01']
    tested_models = ['vit_base_patch16_224', 'vit_base_patch32_224', 't2t_vit_14']
    vit_versions = ['doexp5', 'donegexp025l']
    t2t_versions = ['t', 'p', 't_doexp05l', 't_donegexp05l', 't_donegexp025l']
    train_path = 'output/train/'
    ext = '/model_best.pth.tar'
    if version < 0 or version >= len(tested_models):
        print("Error: Version asked does not exist.")
        return

    if 't2t' not in tested_models[version]:
        if ckpt in vit_versions:
            ckpt_path = train_path + tested_models[version] + '_' + ckpt
            exp_name = tested_models[version] + '_' + ckpt
            model_name = 'custom_' + tested_models[version]
        else:
            ckpt_path = train_path + tested_models[version]
            exp_name = tested_models[version] + ('_pretrained' if p else '_scratch')
            model_name = tested_models[version]
    elif ckpt in t2t_versions:
        ckpt_path = train_path + tested_models[version] + '_' + ckpt
        exp_name = tested_models[version] + '_' + ckpt
        if ckpt in ['t', 'p']:
            model_name = tested_models[version] + '_' + ckpt
        else:
            model_name = 'custom_' + tested_models[version] + '_' + ckpt.split('_')[0]
    else:
        return

    ckpt_file = ckpt_path + ext
    if not p and not os.path.exists(ckpt_file):
        return
    print('\n\t======Starting evaluation of ' + exp_name + '======')
    if 't2t' not in model_name:
        if p:
            model = timm.create_model(model_name, pretrained=True)
        else:
            model = timm.create_model(model_name, checkpoint_path=ckpt_file)
    else:
        if ckpt in ['t', 'p']:
            model = models.T2T.load_t2t_vit(model_name, ckpt_file)
        else:
            model = models.Custom_T2T.load_custom_t2t_vit(model_name, ckpt_file)
    model = model.cuda()

    if os.path.exists(ckpt_path) or p:
        loader = get_val_loader(data, batch_size=5)
        random_index = np.random.randint(0, 10)
        val_path = 'output/CKA_detect/' + exp_name
        if not os.path.exists(val_path):
            os.mkdir(val_path)
        val_path = val_path + '/setup_' + attacks_name[adv_atk] + '_' + attacks_name[detect_atk]
        if not os.path.exists(val_path):
            os.mkdir(val_path)
            os.mkdir(val_path + '/correct_preds/')
            os.mkdir(val_path + '/incorrect_preds/')

        loss_fn = CrossEntropyLoss().cuda()

        im_id = 0
        for it, (input, target) in enumerate(loader):
            if it % 10 == random_index:
                input = input.cuda()
                output_1, sim_mat_1 = prediction_step(model, exp_name, input, target, loss_fn, attacks[adv_atk])
                output_2, sim_mat_2 = prediction_step(model, exp_name, input, target, loss_fn, attacks[detect_atk])
                output_1, output_2 = output_1.detach().cpu().numpy(), output_2.detach().cpu().numpy()
                target = target.detach().cpu().numpy()
                print_simmat_var(sim_mat_1, sim_mat_2, output_1, output_2, target, val_path, im_id, model_name,
                                 attacks_name[adv_atk], attacks_name[detect_atk])
                im_id += 1
            else:
                continue


if __name__ == '__main__':
    main()
