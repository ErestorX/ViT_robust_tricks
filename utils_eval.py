from torch.nn.parallel import DistributedDataParallel as NativeDDP
from models.Custom_T2T import load_custom_t2t_vit
from torch.utils.tensorboard import SummaryWriter
from matplotlib.ticker import PercentFormatter
from models.T2T import load_t2t_vit
from collections import OrderedDict
import matplotlib.pyplot as plt
from torchmetrics import Metric
from einops import rearrange
from timm.utils import *
import numpy as np
import os.path
import torch
import json
import timm
import time
import copy
import cv2


def centered_gram(X):
    def gram(X):
        # ensure correct input shape
        X = rearrange(X, 'b ... -> b (...)')
        return X @ X.T

    def centering_mat(n):
        v_i = torch.ones(n, 1).cuda()
        H = torch.eye(n).cuda() - (v_i @ v_i.T) / n
        return H

    K = gram(X)
    m = K.shape[0]
    H = centering_mat(m)
    return H @ K @ H


def unbiased_hsic_xy(X, Y):
    n = X.shape[0]
    assert n > 3
    v_i = torch.ones(n, 1).cuda()
    K = centered_gram(X)
    L = centered_gram(Y)
    KL = K @ L
    iK = v_i.T @ K
    Li = L @ v_i
    iKi = iK @ v_i
    iLi = v_i.T @ Li

    a = torch.trace(KL)
    b = iKi * iLi / ((n - 1) * (n - 2))
    c = iK @ Li * 2 / (n - 2)

    outv = (a + b - c) / (n * (n - 3))
    return outv.long().item()


class MinibatchCKA(Metric):
    def __init__(self, dist_sync_on_step=False):
        """
        Introduced in: https://arxiv.org/pdf/2010.15327.pdf
        Implemented to reproduce the results in: https://arxiv.org/pdf/2108.08810v1.pdf
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("_xx", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("_xy", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("_yy", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, X: torch.Tensor, Y: torch.Tensor):
        # NB: torchmetrics Bootstrap resampling janks up batch shape by varying number of samples per batch
        self._xx += unbiased_hsic_xy(X, X)
        self._xy += unbiased_hsic_xy(X, Y)
        self._yy += unbiased_hsic_xy(Y, Y)

    def compute(self):
        xx, xy, yy = self._xx, self._xy, self._yy
        return xy / (torch.sqrt(xx) * torch.sqrt(yy))


def get_all_hooks(model, is_t2t=False, is_performer=False):
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
            hook = HookedCache(model.module, tgt)
            hooks.append(hook)
        hook = HookedCache(model.module, 'tokens_to_token.project')
        hooks.append(hook)
    for j, block in enumerate(model.module.blocks):
        hook = HookedCache(model.module, 'blocks.{}.attn.qkv'.format(j))
        hooks.append(hook)
        hook = HookedCache(model.module, 'blocks.{}.attn.proj'.format(j))
        hooks.append(hook)
        hook = HookedCache(model.module, 'blocks.{}.mlp'.format(j))
        hooks.append(hook)
    return hooks


class HookedCache:
    def __init__(self, model, target):
        self.model = model
        self.target = target
        self.clear()
        self._extract_target()
        self._register_hook()

    @property
    def value(self):
        return self._cache

    def clear(self):
        self._cache = None

    def _extract_target(self):
        for name, module in self.model.named_modules():
            if name == self.target:
                self._target = module
                return
        raise ValueError('Target {} not found in model'.format(self.target))

    def _register_hook(self):
        def _hook(module, in_val, out_val):
            self._cache = out_val

        self._target.register_forward_hook(_hook)


def get_simmat_from_metrics(metrics):
    vals = []
    for i, ckas in enumerate(metrics):
        for j, cka in enumerate(ckas):
            z = cka.compute().item()
            vals.append((i, j, z))
    sim_mat = torch.zeros(i + 1, j + 1)
    for i, j, z in vals:
        sim_mat[i, j] = z
    return sim_mat


def make_pairwise_metrics(mod1_hooks, mod2_hooks):
    metrics = []
    for _ in mod1_hooks:
        metrics.append([])
        for _ in mod2_hooks:
            metrics[-1].append(MinibatchCKA().cuda())
    return metrics


def update_metrics(mod1_hooks, mod2_hooks, metrics, metric_name, it, writer, do_log):
    for i, hook1 in enumerate(mod1_hooks):
        for j, hook2 in enumerate(mod2_hooks):
            cka = metrics[i][j]
            X, Y = hook1.value, hook2.value
            cka.update(X, Y)
            if 0 in (i, j):
                _metric_name = f"{metric_name}_{i}-{j}"
                v = cka.compute()
                writer.add_scalar(_metric_name, v, it)
    if do_log:
        sim_mat = get_simmat_from_metrics(metrics)
        sim_mat = sim_mat.unsqueeze(0) * 255
        writer.add_image(metric_name, sim_mat, it)


def torch_weighted_mean(tensor, weights):
    return (tensor * weights).sum((1, 2, 3)) / weights.sum((1, 2, 3))


def attn_distance(model, name_model, loader, summary, args):
    if 'AttDist_cln' not in summary.keys():
        if args.local_rank == 0:
            print('\t---Starting clean attention distance computation---')
        def get_features(name):
            def hook(model, input, output):
                qkvs[name] = output.detach()

            return hook

        index = 0
        t2t = 'T2T' in name_model
        performer = t2t and '-p' in name_model
        if t2t:
            index = 2
            if performer:
                model.module.tokens_to_token.attention1.kqv.register_forward_hook(get_features('0'))
                model.module.tokens_to_token.attention2.kqv.register_forward_hook(get_features('1'))
            else:
                model.module.tokens_to_token.attention1.attn.qkv.register_forward_hook(get_features('0'))
                model.module.tokens_to_token.attention2.attn.qkv.register_forward_hook(get_features('1'))

        for block_id, block in enumerate(model.module.blocks):
            block.attn.qkv.register_forward_hook(get_features(str(block_id + index)))

        patch_size = 32 if '32' in name_model else 16
        model.eval()
        qkvs = {}
        statistics = {}
        for batch_idx, (input, target) in enumerate(loader):
            _ = model(input)
            for block, qkv in qkvs.items():
                if t2t and int(block) < 2:
                    num_heads = 1
                else:
                    num_heads = model.module.blocks[0].attn.num_heads
                B, N, CCC = qkv.shape
                C = CCC // 3
                if performer and int(block) < 2:
                    if int(block) == 0:
                        k, q, v = torch.split(qkv, model.module.tokens_to_token.attention1.emb, dim=-1)
                        k, q = model.module.tokens_to_token.attention1.prm_exp(k), model.module.tokens_to_token.attention1.prm_exp(q)
                    elif int(block) == 1:
                        k, q, v = torch.split(qkv, model.module.tokens_to_token.attention2.emb, dim=-1)
                        k, q = model.module.tokens_to_token.attention2.prm_exp(k), model.module.tokens_to_token.attention2.prm_exp(q)
                    shape_k, shape_q = k.shape, q.shape
                    k, q = k.reshape(shape_k[0], 1, shape_k[1], shape_k[2]), q.reshape(shape_q[0], 1, shape_q[1], shape_q[2])
                else:
                    qkv = qkv.reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
                    q, k, _ = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
                attn = (q @ k.transpose(-2, -1)) * (num_heads ** -0.5)
                attn = attn.softmax(dim=-1).permute(1, 0, 2, 3)
                N = N-1
                attn = attn[:, :, 1:, 1:]
                vect = torch.arange(N).reshape((1, N))
                dist_map = torch.sqrt((torch.abs((vect - torch.transpose(vect, 0, 1))) % N ** 0.5) ** 2 + ((vect - torch.transpose(vect, 0, 1)) // N ** 0.5) ** 2)
                if t2t and int(block) == 0:
                    patch_size = patch_size/4
                elif t2t and int(block) == 1:
                    patch_size = patch_size/2
                val = torch.as_tensor(dist_map).cuda().unsqueeze(0).repeat(num_heads, B, 1, 1) * patch_size
                weights = attn
                avg = torch_weighted_mean(val, weights)
                std = torch.sqrt(torch_weighted_mean((val - avg.reshape(num_heads, 1, 1, 1)) ** 2, weights))
                if block not in statistics.keys():
                    statistics[block] = {'avgHead': avg.cpu().numpy().tolist(), 'stdHead': std.cpu().numpy().tolist()}
                else:
                    statistics[block]['avgHead'] = (np.add(np.asarray(statistics[block]['avgHead']) * batch_idx,
                                                           avg.cpu().numpy()) / (batch_idx + 1)).tolist()
                    statistics[block]['stdHead'] = (np.add(np.asarray(statistics[block]['stdHead']) * batch_idx,
                                                           std.cpu().numpy()) / (batch_idx + 1)).tolist()
        summary['AttDist_cln'] = list(statistics.values())


def adv_attn_distance(model, name_model, loss_fn, loader, summary, args, epsilonMax=.062, pgd_steps=1, step_size=1):
    key = '_'.join(['AttDist_adv', 'steps:' + str(pgd_steps), 'eps:' + str(epsilonMax)])
    if key not in summary.keys():
        if args.local_rank == 0:
            print('\t---Starting adversarial attention distance computation---')
        def get_features(name):
            def hook(model, input, output):
                qkvs[name] = output.detach()

            return hook

        index = 0
        t2t = 'T2T' in name_model
        performer = t2t and '-p' in name_model
        if t2t:
            index = 2
            if performer:
                model.module.tokens_to_token.attention1.kqv.register_forward_hook(get_features('0'))
                model.module.tokens_to_token.attention2.kqv.register_forward_hook(get_features('1'))
            else:
                model.module.tokens_to_token.attention1.attn.qkv.register_forward_hook(get_features('0'))
                model.module.tokens_to_token.attention2.attn.qkv.register_forward_hook(get_features('1'))

        for block_id, block in enumerate(model.module.blocks):
            block.attn.qkv.register_forward_hook(get_features(str(block_id + index)))

        patch_size = 32 if '32' in name_model else 16
        model.eval()
        qkvs = {}
        statistics = {}
        for batch_idx, (input, target) in enumerate(loader):
            input_orig = input.clone()
            for i in range(pgd_steps):
                input.requires_grad = True
                output = model(input)
                cost = loss_fn(output, target)
                grad = torch.autograd.grad(cost, input, retain_graph=False, create_graph=False)[0]
                input = input + step_size * grad.sign()
                input = input_orig + torch.clamp(input - input_orig, -epsilonMax, epsilonMax)
                input = torch.clamp(input, -1, 1).detach()
            _ = model(input)
            for block, qkv in qkvs.items():
                if t2t and int(block) < 2:
                    num_heads = 1
                else:
                    num_heads = model.module.blocks[0].attn.num_heads
                B, N, CCC = qkv.shape
                C = CCC // 3
                if performer and int(block) < 2:
                    if int(block) == 0:
                        k, q, v = torch.split(qkv, model.module.tokens_to_token.attention1.emb, dim=-1)
                        k, q = model.module.tokens_to_token.attention1.prm_exp(k), model.module.tokens_to_token.attention1.prm_exp(q)
                    elif int(block) == 1:
                        k, q, v = torch.split(qkv, model.module.tokens_to_token.attention2.emb, dim=-1)
                        k, q = model.module.tokens_to_token.attention2.prm_exp(k), model.module.tokens_to_token.attention2.prm_exp(q)
                    shape_k, shape_q = k.shape, q.shape
                    k, q = k.reshape(shape_k[0], 1, shape_k[1], shape_k[2]), q.reshape(shape_q[0], 1, shape_q[1], shape_q[2])
                else:
                    qkv = qkv.reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
                    q, k, _ = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
                attn = (q @ k.transpose(-2, -1)) * (num_heads ** -0.5)
                attn = attn.softmax(dim=-1)
                _, H, _, _ = attn.shape
                attn = attn.permute(1, 0, 2, 3)
                N = N - 1
                attn = attn[:, :, 1:, 1:]
                vect = torch.arange(N).reshape((1, N))
                dist_map = torch.sqrt((torch.abs((vect - torch.transpose(vect, 0, 1))) % N ** 0.5) ** 2 + ((vect - torch.transpose(vect, 0, 1)) // N ** 0.5) ** 2)
                if t2t and int(block) == 0:
                    patch_size = patch_size / 4
                elif t2t and int(block) == 1:
                    patch_size = patch_size / 2

                val = torch.as_tensor(dist_map).cuda().unsqueeze(0).repeat(num_heads, B, 1, 1) * patch_size
                weights = attn
                avg = torch_weighted_mean(val, weights)
                std = torch.sqrt(torch_weighted_mean((val - avg.reshape(num_heads, 1, 1, 1)) ** 2, weights))
                if block not in statistics.keys():
                    statistics[block] = {'avgHead': avg.cpu().numpy().tolist(), 'stdHead': std.cpu().numpy().tolist()}
                else:
                    statistics[block]['avgHead'] = (np.add(np.asarray(statistics[block]['avgHead']) * batch_idx,
                                                          avg.cpu().numpy()) / (batch_idx + 1)).tolist()
                    statistics[block]['stdHead'] = (np.add(np.asarray(statistics[block]['stdHead']) * batch_idx,
                                                          std.cpu().numpy()) / (batch_idx + 1)).tolist()
        summary[key] = list(statistics.values())


def freq_hist(title, val_path):
    for img in ['clean', 'adv', 'perturb']:
        image = cv2.imread(val_path + '/' + img + '_batch.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        data = image.ravel()

        _ = plt.hist(data, weights=np.ones(len(data)) / len(data), bins=256, color='orange', )
        _ = plt.title(title)
        _ = plt.xlabel('Intensity Value')
        _ = plt.ylabel('Count')
        _ = plt.ylim(top=.2 if img == 'perturb' else .01)
        _ = plt.legend(['Total'])
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.savefig(val_path + '/Pixel_hist_' + img + '.png')
        plt.close()
        plt.imsave(val_path + '/Freq_' + img + '.png', np.log(abs(np.fft.fftshift(np.fft.fft2(image)))), cmap='gray')


def get_clean_CKA(json_summaries, model_t, model_t_name, model_c, model_c_name, data_loader, args):
    if model_c_name not in json_summaries.keys():
        if args.local_rank == 0:
            print('\t---Starting clean CKA computation with ' + model_c_name + '---')
        writer = SummaryWriter()
        modc_hooks = get_all_hooks(model_c, is_t2t='t2t' in model_c_name, is_performer=model_c_name.split('_')[3] == 'p')
        modt_hooks = get_all_hooks(model_t, is_t2t='t2t' in model_t_name, is_performer=model_t_name.split('_')[3] == 'p')
        metrics_ct = make_pairwise_metrics(modc_hooks, modt_hooks)

        with torch.no_grad():
            for it, (input, target) in enumerate(data_loader):
                do_log = (it % 10 == 0)
                _ = model_c(input)
                _ = model_t(input)
                update_metrics(modc_hooks, modt_hooks, metrics_ct, "cka/ct", it, writer, do_log)
                for hook0 in modc_hooks:
                    for hook1 in modt_hooks:
                        hook0.clear()
                        hook1.clear()

        sim_mat = get_simmat_from_metrics(metrics_ct)
        json_summaries[model_c_name] = sim_mat.tolist()


def get_transfer_CKA(json_summaries, model_t, model_t_name, model_c, model_c_name, data_loader, loss_fn, args, epsilonMax=0.062, pgd_steps=1, step_size=1):
    if model_c_name not in json_summaries.keys():
        if args.local_rank == 0:
            print('\t---Starting transfer CKA computation with ' + model_c_name + '---')
        writer = SummaryWriter()
        modc_hooks = get_all_hooks(model_c, is_t2t='t2t' in model_c_name, is_performer=model_c_name.split('_')[3] == 'p')
        modt_hooks = get_all_hooks(model_t, is_t2t='t2t' in model_t_name, is_performer=model_t_name.split('_')[3] == 'p')
        metrics_ct = make_pairwise_metrics(modc_hooks, modt_hooks)

        for it, (input, target) in enumerate(data_loader):
            input_orig = input.clone()
            do_log = (it % 10 == 0)
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
            _ = model_t(input)
            update_metrics(modc_hooks, modt_hooks, metrics_ct, "cka/ct", it, writer, do_log)
            for hook0 in modc_hooks:
                for hook1 in modt_hooks:
                    hook0.clear()
                    hook1.clear()

        sim_mat = get_simmat_from_metrics(metrics_ct)
        json_summaries[model_c_name] = sim_mat.tolist()


def get_adversarial_CKA(json_summaries, model_t, model_t_name, data_loader, loss_fn, args, epsilonMax=0.062, pgd_steps=1, step_size=1):
    key = '_'.join(['CKA_adv', 'steps:' + str(pgd_steps), 'eps:' + str(epsilonMax)])
    if key not in json_summaries.keys():
        if args.local_rank == 0:
            print('\t---Starting adversarial CKA computation---')
        model_c = copy.deepcopy(model_t)
        writer = SummaryWriter()
        modc_hooks = get_all_hooks(model_c, is_t2t='t2t' in model_t_name, is_performer=model_t_name.split('_')[3] == 'p')
        modt_hooks = get_all_hooks(model_t, is_t2t='t2t' in model_t_name, is_performer=model_t_name.split('_')[3] == 'p')
        metrics_ct = make_pairwise_metrics(modc_hooks, modt_hooks)

        for it, (input, target) in enumerate(data_loader):
            input_orig = input.clone()
            do_log = (it % 10 == 0)
            _ = model_t(input)
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
            update_metrics(modc_hooks, modt_hooks, metrics_ct, "cka/ct", it, writer, do_log)
            for hook0 in modc_hooks:
                for hook1 in modt_hooks:
                    hook0.clear()
                    hook1.clear()

        sim_mat = get_simmat_from_metrics(metrics_ct)
        json_summaries[key] = sim_mat.tolist()


def get_clean_CKA_single_element(json_summaries, model_t, model_t_name, model_c, model_c_name, data_loader, args):
    if model_c_name not in json_summaries.keys():
        if args.local_rank == 0:
            print('\t---Starting clean CKA computation on single element with ' + model_c_name + '---')
        writer = SummaryWriter()
        modc_hooks = get_all_hooks(model_c, is_t2t='t2t' in model_c_name, is_performer=model_c_name.split('_')[3] == 'p')
        modt_hooks = get_all_hooks(model_t, is_t2t='t2t' in model_t_name, is_performer=model_t_name.split('_')[3] == 'p')
        metrics_ct = make_pairwise_metrics(modc_hooks, modt_hooks)

        with torch.no_grad():
            for it, (input, target) in enumerate(data_loader):
                input = input[:4]
                do_log = (it % 10 == 0)
                _ = model_c(input)
                _ = model_t(input)
                update_metrics(modc_hooks, modt_hooks, metrics_ct, "cka/ct", it, writer, do_log)
                for hook0 in modc_hooks:
                    for hook1 in modt_hooks:
                        hook0.clear()
                        hook1.clear()
                break

        sim_mat = get_simmat_from_metrics(metrics_ct)
        json_summaries[model_c_name] = sim_mat.tolist()


def get_transfer_CKA_single_element(json_summaries, model_t, model_t_name, model_c, model_c_name, data_loader, loss_fn, args, epsilonMax=0.062, pgd_steps=1, step_size=1):
    if model_c_name not in json_summaries.keys():
        if args.local_rank == 0:
            print('\t---Starting transfer CKA computation on single element with ' + model_c_name + '---')
        writer = SummaryWriter()
        modc_hooks = get_all_hooks(model_c, is_t2t='t2t' in model_c_name, is_performer=model_c_name.split('_')[3] == 'p')
        modt_hooks = get_all_hooks(model_t, is_t2t='t2t' in model_t_name, is_performer=model_t_name.split('_')[3] == 'p')
        metrics_ct = make_pairwise_metrics(modc_hooks, modt_hooks)

        for it, (input, target) in enumerate(data_loader):
            input, target = input[:4], target[:4]
            input_orig = input.clone()
            do_log = (it % 10 == 0)
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
            _ = model_t(input)
            update_metrics(modc_hooks, modt_hooks, metrics_ct, "cka/ct", it, writer, do_log)
            for hook0 in modc_hooks:
                for hook1 in modt_hooks:
                    hook0.clear()
                    hook1.clear()
            break

        sim_mat = get_simmat_from_metrics(metrics_ct)
        json_summaries[model_c_name] = sim_mat.tolist()


def get_adversarial_CKA_single_element(json_summaries, model_t, model_t_name, data_loader, loss_fn, args, epsilonMax=0.062, pgd_steps=1, step_size=1):
    key = '_'.join(['CKA_single_adv', 'steps:' + str(pgd_steps), 'eps:' + str(epsilonMax)])
    if key not in json_summaries.keys():
        if args.local_rank == 0:
            print('\t---Starting adversarial CKA computation on single element---')
        model_c = copy.deepcopy(model_t)
        writer = SummaryWriter()
        modc_hooks = get_all_hooks(model_c, is_t2t='t2t' in model_t_name, is_performer=model_t_name.split('_')[3] == 'p')
        modt_hooks = get_all_hooks(model_t, is_t2t='t2t' in model_t_name, is_performer=model_t_name.split('_')[3] == 'p')
        metrics_ct = make_pairwise_metrics(modc_hooks, modt_hooks)

        for it, (input, target) in enumerate(data_loader):
            input, target = input[:4], target[:4]
            input_orig = input.clone()
            do_log = (it % 10 == 0)
            _ = model_t(input)
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
            update_metrics(modc_hooks, modt_hooks, metrics_ct, "cka/ct", it, writer, do_log)
            for hook0 in modc_hooks:
                for hook1 in modt_hooks:
                    hook0.clear()
                    hook1.clear()
            break

        sim_mat = get_simmat_from_metrics(metrics_ct)
        json_summaries[key] = sim_mat.tolist()


def get_CKAs(json_summaries, model_1, name_model_1, model_2, name_model_2, loader, loss_fn, args, model_2_ckpt_file='', pretrained=False, epsilonMax=0.062, pgd_steps=1, step_size=1):
    if "CKA_cln" not in json_summaries.keys():
        json_summaries["CKA_cln"] = {}
    trf_key = '_'.join(['CKA_trf', 'steps:' + str(pgd_steps), 'eps:' + str(epsilonMax)])
    if trf_key not in json_summaries.keys():
        json_summaries[trf_key] = {}
    if 't2t' in model_2:
        if "custom" in model_2:
            model_2 = load_custom_t2t_vit(model_2, model_2_ckpt_file)
        else:
            model_2 = load_t2t_vit(model_2, model_2_ckpt_file)
    elif not pretrained:
        model_2 = timm.create_model(model_2, checkpoint_path=model_2_ckpt_file)
    else:
        model_2 = timm.create_model(model_2, pretrained=True)
    model_2 = model_2.cuda()
    if args.distributed:
        model_2 = NativeDDP(model_2, device_ids=[args.local_rank])

    get_clean_CKA(json_summaries["CKA_cln"], model_1, name_model_1, model_2, name_model_2, loader, args)
    get_adversarial_CKA(json_summaries, model_1, name_model_1, loader, loss_fn, args, epsilonMax, pgd_steps, step_size=step_size)
    get_transfer_CKA(json_summaries[trf_key], model_1, name_model_1, model_2, name_model_2, loader, loss_fn, args, epsilonMax, pgd_steps, step_size=step_size)


def get_CKAs_single_element(json_summaries, model_1, name_model_1, model_2, name_model_2, loader, loss_fn, args, model_2_ckpt_file='', pretrained=False, epsilonMax=0.062, pgd_steps=1, step_size=1):
    if "CKA_single_cln" not in json_summaries.keys():
        json_summaries["CKA_single_cln"] = {}
    trf_key = '_'.join(['CKA_single_trf', 'steps:' + str(pgd_steps), 'eps:' + str(epsilonMax)])
    if trf_key not in json_summaries.keys():
        json_summaries[trf_key] = {}
    if 't2t' in model_2:
        if "custom" in model_2:
            model_2 = load_custom_t2t_vit(model_2, model_2_ckpt_file)
        else:
            model_2 = load_t2t_vit(model_2, model_2_ckpt_file)
    elif not pretrained:
        model_2 = timm.create_model(model_2, checkpoint_path=model_2_ckpt_file)
    else:
        model_2 = timm.create_model(model_2, pretrained=True)
    model_2 = model_2.cuda()
    if args.distributed:
        model_2 = NativeDDP(model_2, device_ids=[args.local_rank])

    get_clean_CKA_single_element(json_summaries["CKA_single_cln"], model_1, name_model_1, model_2, name_model_2, loader, args)
    get_adversarial_CKA_single_element(json_summaries, model_1, name_model_1, loader, loss_fn, args, epsilonMax, pgd_steps, step_size=step_size)
    get_transfer_CKA_single_element(json_summaries[trf_key], model_1, name_model_1, model_2, name_model_2, loader, loss_fn, args, epsilonMax, pgd_steps, step_size=step_size)


def save_experiment_results(json_file, data, local_rank):
    if local_rank == 0:
        with open(json_file, 'w') as f:
            json.dump(data, f)


def do_all_CKAs(CKA_fn, all_summaries, json_file, model, exp_name, loader, loss_fn, tested_models, vit_versions, t2t_versions, train_path, ext, args):
    for model_name_2 in tested_models:
        if 't2t' not in model_name_2:
            for version in vit_versions:
                ckpt_file = train_path + model_name_2 + '_' + version + ext
                if os.path.exists(ckpt_file):
                    CKA_fn(all_summaries[exp_name], model, exp_name, 'custom_' + model_name_2,
                             model_name_2 + '_' + version, loader, loss_fn, args, model_2_ckpt_file=ckpt_file,
                             epsilonMax=args.epsilon, pgd_steps=args.steps, step_size=args.step_size)
                    save_experiment_results(json_file, all_summaries, args.local_rank)
            ckpt_file = train_path + model_name_2 + ext
            if os.path.exists(ckpt_file):
                CKA_fn(all_summaries[exp_name], model, exp_name, model_name_2, model_name_2 + '_scratch', loader,
                         loss_fn, args, model_2_ckpt_file=ckpt_file, epsilonMax=args.epsilon, pgd_steps=args.steps, step_size=args.step_size)
                save_experiment_results(json_file, all_summaries, args.local_rank)
            CKA_fn(all_summaries[exp_name], model, exp_name, model_name_2, model_name_2 + '_pretrained', loader,
                     loss_fn, args, pretrained=True, epsilonMax=args.epsilon, pgd_steps=args.steps, step_size=args.step_size)
            save_experiment_results(json_file, all_summaries, args.local_rank)
        else:
            for version in t2t_versions:
                ckpt_file = train_path + model_name_2 + '_' + version + ext
                if os.path.exists(ckpt_file):
                    if version in ['p', 't']:
                        model_type = model_name_2 + '_' + version
                    else:
                        model_type = 'custom_' + model_name_2 + '_' + '_'.join(version.split('_')[:-1])
                    CKA_fn(all_summaries[exp_name], model, exp_name, model_type, model_name_2 + '_' + version,
                             loader, loss_fn, args, model_2_ckpt_file=ckpt_file, epsilonMax=args.epsilon,
                             pgd_steps=args.steps, step_size=args.step_size)
                    save_experiment_results(json_file, all_summaries, args.local_rank)


def get_val_loader(data_path, batch_size=64):
    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
    dataset_eval = timm.data.create_dataset('', root=data_path, split='validation', is_training=False, batch_size=batch_size)
    loader_eval = timm.data.create_loader(
        dataset_eval,
        input_size=[3, 224, 224],
        batch_size=batch_size,
        is_training=False,
        use_prefetcher=True,
        interpolation='bicubic',
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        num_workers=4,
        distributed=distributed,
        crop_pct=0.9,
        pin_memory=False,
    )
    return loader_eval


def validate(model, loader, loss_fn, summary, args):
    if 'Metrics_cln' not in summary.keys():
        if args.local_rank == 0:
            print('\t---Starting validation on clean DS---')
        batch_time_m = AverageMeter()
        losses_m = AverageMeter()
        top1_m = AverageMeter()
        top5_m = AverageMeter()
        model.eval()
        end = time.time()
        last_idx = len(loader) - 1

        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(loader):
                last_batch = batch_idx == last_idx
                output = model(input)
                if isinstance(output, (tuple, list)):
                    output = output[0]

                loss = loss_fn(output, target)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                if args.distributed:
                    reduced_loss = reduce_tensor(loss.data, args.world_size)
                    acc1 = reduce_tensor(acc1, args.world_size)
                    acc5 = reduce_tensor(acc5, args.world_size)
                else:
                    reduced_loss = loss.data
                torch.cuda.synchronize()
                losses_m.update(reduced_loss.item(), input.size(0))
                top1_m.update(acc1.item(), output.size(0))
                top5_m.update(acc5.item(), output.size(0))

                batch_time_m.update(time.time() - end)
                end = time.time()
                if (last_batch or batch_idx % 100 == 0) and args.local_rank == 0:
                    log_name = 'Clean'
                    print('{0}: [{1:>4d}/{2}]  Acc@1: {top1.avg:>7.4f}'.format(log_name, batch_idx, last_idx,
                                                                               batch_time=batch_time_m, top1=top1_m))

        summary['Metrics_cln'] = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])


def validate_attack(model, loader, loss_fn, summary, args, epsilonMax=0.062, pgd_steps=1, step_size=1):
    key = '_'.join(['Metrics_adv', 'steps:'+str(pgd_steps), 'eps:'+str(epsilonMax)])
    if key not in summary.keys():
        if args.local_rank == 0:
            print('\t---Starting validation on attacked DS---')
        batch_time_m = AverageMeter()
        losses_m = AverageMeter()
        top1_m = AverageMeter()
        top5_m = AverageMeter()
        model.eval()
        end = time.time()
        last_idx = len(loader) - 1

        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            input_orig = input.clone()
            for _ in range(pgd_steps):
                input.requires_grad = True
                output = model(input)
                model.zero_grad()
                cost = loss_fn(output, target)
                grad = torch.autograd.grad(cost, input, retain_graph=False, create_graph=False)[0]
                input = input + step_size * grad.sign()
                input = input_orig + torch.clamp(input - input_orig, -epsilonMax, epsilonMax)
                input = torch.clamp(input, -1, 1).detach()
            output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data
            torch.cuda.synchronize()
            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if (last_batch or batch_idx % 100 == 0 )and args.local_rank == 0:
                log_name = 'Adversarial'
                print('{0}: [{1:>4d}/{2}]  Acc@1: {top1.avg:>7.4f}'.format(log_name, batch_idx, last_idx,
                                                                           batch_time=batch_time_m, top1=top1_m))
        summary[key] = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])


def update_attention_stat(model, qkvs, statistics, batch_idx, patch_size, t2t, performer):
    for block, qkv in qkvs.items():
        if t2t and int(block) < 2:
            num_heads = 1
        else:
            num_heads = model.module.blocks[0].attn.num_heads
        B, N, CCC = qkv.shape
        C = CCC // 3
        if performer and int(block) < 2:
            if int(block) == 0:
                k, q, v = torch.split(qkv, model.module.tokens_to_token.attention1.emb, dim=-1)
                k, q = model.module.tokens_to_token.attention1.prm_exp(
                    k), model.module.tokens_to_token.attention1.prm_exp(q)
            elif int(block) == 1:
                k, q, v = torch.split(qkv, model.module.tokens_to_token.attention2.emb, dim=-1)
                k, q = model.module.tokens_to_token.attention2.prm_exp(
                    k), model.module.tokens_to_token.attention2.prm_exp(q)
            shape_k, shape_q = k.shape, q.shape
            k, q = k.reshape(shape_k[0], 1, shape_k[1], shape_k[2]), q.reshape(shape_q[0], 1, shape_q[1],
                                                                               shape_q[2])
        else:
            qkv = qkv.reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * (num_heads ** -0.5)
        attn = attn.softmax(dim=-1).permute(1, 0, 2, 3)
        N = N - 1
        attn = attn[:, :, 1:, 1:]
        vect = torch.arange(N).reshape((1, N))
        dist_map = torch.sqrt((torch.abs((vect - torch.transpose(vect, 0, 1))) % N ** 0.5) ** 2 + (
                (vect - torch.transpose(vect, 0, 1)) // N ** 0.5) ** 2)
        if t2t and int(block) == 0:
            patch_size = patch_size / 4
        elif t2t and int(block) == 1:
            patch_size = patch_size / 2
        val = torch.as_tensor(dist_map).cuda().unsqueeze(0).repeat(num_heads, B, 1, 1) * patch_size
        weights = attn
        avg = torch_weighted_mean(val, weights)
        std = torch.sqrt(torch_weighted_mean((val - avg.reshape(num_heads, 1, 1, 1)) ** 2, weights))
        if block not in statistics.keys():
            statistics[block] = {'avgHead': avg.cpu().numpy().tolist(),
                                 'stdHead': std.cpu().numpy().tolist()}
        else:
            statistics[block]['avgHead'] = (np.add(np.asarray(statistics[block]['avgHead']) * batch_idx,
                                                   avg.cpu().numpy()) / (batch_idx + 1)).tolist()
            statistics[block]['stdHead'] = (np.add(np.asarray(statistics[block]['stdHead']) * batch_idx,
                                                   std.cpu().numpy()) / (batch_idx + 1)).tolist()
    return statistics


def get_accuracy_and_attention(model, name_model, loader, loss_fn, summary, args):
    if 'Metrics_cln' not in summary.keys() or 'AttDist_cln' not in summary.keys():
        if args.local_rank == 0:
            print('\t---Starting validation on clean DS---')
        batch_time_m = AverageMeter()
        losses_m = AverageMeter()
        top1_m = AverageMeter()
        top5_m = AverageMeter()

        def get_features(name):
            def hook(model, input, output):
                qkvs[name] = output.detach()

            return hook

        index = 0
        t2t = 'T2T' in name_model
        performer = t2t and '-p' in name_model
        if t2t:
            index = 2
            if performer:
                model.module.tokens_to_token.attention1.kqv.register_forward_hook(get_features('0'))
                model.module.tokens_to_token.attention2.kqv.register_forward_hook(get_features('1'))
            else:
                model.module.tokens_to_token.attention1.attn.qkv.register_forward_hook(get_features('0'))
                model.module.tokens_to_token.attention2.attn.qkv.register_forward_hook(get_features('1'))

        for block_id, block in enumerate(model.module.blocks):
            block.attn.qkv.register_forward_hook(get_features(str(block_id + index)))

        patch_size = 32 if '32' in name_model else 16
        model.eval()
        qkvs = {}
        statistics = {}
        end = time.time()
        last_idx = len(loader) - 1

        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            output = model(input)
            statistics = update_attention_stat(model, qkvs, statistics, batch_idx, patch_size, t2t, performer)
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data
            torch.cuda.synchronize()
            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if (last_batch or batch_idx % 100 == 0) and args.local_rank == 0:
                log_name = 'Clean'
                print('{0}: [{1:>4d}/{2}]  Acc@1: {top1.avg:>7.4f}'.format(log_name, batch_idx, last_idx,
                                                                           batch_time=batch_time_m,
                                                                           top1=top1_m))

        summary['Metrics_cln'] = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])
        summary['AttDist_cln'] = list(statistics.values())
        avgHead = []
        stdHead = []
        for block in summary['AttDist_cln']:
            avgHead.append(block['avgHead'])
            stdHead.append(block['stdHead'])
        summary['AttDist_cln'] = {'avgHead': avgHead, 'stdHead': stdHead}


def get_attack_accuracy_and_attention(model, name_model, loader, loss_fn, summary, args, epsilonMax=0.062, pgd_steps=1, step_size=1):
    key_metrics = '_'.join(['Metrics_adv', 'steps:' + str(pgd_steps), 'eps:' + str(epsilonMax)])
    key_attDist = '_'.join(['AttDist_adv', 'steps:' + str(pgd_steps), 'eps:' + str(epsilonMax)])
    if key_metrics not in summary.keys() or key_attDist not in summary.keys():
        if args.local_rank == 0:
            print('\t---Starting validation on DS attacked: steps={} epsilon={}---'.format(pgd_steps, epsilonMax))
        batch_time_m = AverageMeter()
        losses_m = AverageMeter()
        top1_m = AverageMeter()
        top5_m = AverageMeter()

        def get_features(name):
            def hook(model, input, output):
                qkvs[name] = output.detach()

            return hook

        index = 0
        t2t = 'T2T' in name_model
        performer = t2t and '-p' in name_model
        if t2t:
            index = 2
            if performer:
                model.module.tokens_to_token.attention1.kqv.register_forward_hook(get_features('0'))
                model.module.tokens_to_token.attention2.kqv.register_forward_hook(get_features('1'))
            else:
                model.module.tokens_to_token.attention1.attn.qkv.register_forward_hook(get_features('0'))
                model.module.tokens_to_token.attention2.attn.qkv.register_forward_hook(get_features('1'))

        for block_id, block in enumerate(model.module.blocks):
            block.attn.qkv.register_forward_hook(get_features(str(block_id + index)))

        patch_size = 32 if '32' in name_model else 16
        model.eval()
        qkvs = {}
        statistics = {}
        end = time.time()
        last_idx = len(loader) - 1

        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            input_orig = input.clone()
            for _ in range(pgd_steps):
                input.requires_grad = True
                output = model(input)
                model.zero_grad()
                cost = loss_fn(output, target)
                grad = torch.autograd.grad(cost, input, retain_graph=False, create_graph=False)[0]
                input = input + step_size * grad.sign()
                input = input_orig + torch.clamp(input - input_orig, -epsilonMax, epsilonMax)
                input = torch.clamp(input, -1, 1).detach()
            output = model(input)
            statistics = update_attention_stat(model, qkvs, statistics, batch_idx, patch_size, t2t, performer)
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data
            torch.cuda.synchronize()
            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if (last_batch or batch_idx % 100 == 0) and args.local_rank == 0:
                log_name = 'Clean'
                print('{0}: [{1:>4d}/{2}]  Acc@1: {top1.avg:>7.4f}'.format(log_name, batch_idx, last_idx,
                                                                           batch_time=batch_time_m,
                                                                           top1=top1_m))

        summary[key_metrics] = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])
        summary[key_attDist] = list(statistics.values())
        avgHead = []
        stdHead = []
        for block in summary[key_attDist]:
            avgHead.append(block['avgHead'])
            stdHead.append(block['stdHead'])
        summary[key_attDist] = {'avgHead': avgHead, 'stdHead': stdHead}
