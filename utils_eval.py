from torch.utils.tensorboard import SummaryWriter
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
from torchmetrics import Metric
from einops import rearrange
import numpy as np
import torch
import cv2

DEVICE = torch.device('cuda')

def gram(X):
    # ensure correct input shape
    X = rearrange(X, 'b ... -> b (...)')
    return X @ X.T

def centering_mat(n):
    v_i = torch.ones(n,1, device=DEVICE)
    H = torch.eye(n, device=DEVICE) - (v_i @ v_i.T) / n
    return H

def centered_gram(X):
    K = gram(X)
    m = K.shape[0]
    H = centering_mat(m)
    return H @ K @ H

def unbiased_hsic_xy(X,Y):
    n = X.shape[0]
    assert n > 3
    v_i = torch.ones(n,1, device=DEVICE)
    K = centered_gram(X)
    L = centered_gram(Y)
    KL = K @ L
    iK = v_i.T @ K
    Li = L @ v_i
    iKi = iK @ v_i
    iLi = v_i.T @ Li

    a = torch.trace(KL)
    b = iKi * iLi / ((n-1)*(n-2))
    c = iK @ Li * 2 / (n-2)

    outv = (a + b - c) / (n*(n-3))
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
        self._xx += unbiased_hsic_xy(X,X)
        self._xy += unbiased_hsic_xy(X,Y)
        self._yy += unbiased_hsic_xy(Y,Y)
    def compute(self):
        xx, xy, yy = self._xx, self._xy, self._yy
        return xy / (torch.sqrt(xx) * torch.sqrt(yy))


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
  for i_ in mod1_hooks:
    metrics.append([])
    for j_ in mod2_hooks:
      metrics[-1].append(MinibatchCKA().cuda())
  return metrics


def update_metrics(mod1_hooks, mod2_hooks, metrics, metric_name, it, writer):
    for i, hook1 in enumerate(mod1_hooks):
      for j, hook2 in enumerate(mod2_hooks):
        cka = metrics[i][j]
        X,Y = hook1.value, hook2.value
        cka.update(X,Y)
        if 0 in (i,j):
          _metric_name = f"{metric_name}_{i}-{j}"
          v = cka.compute()
          writer.add_scalar(_metric_name, v, it)
    sim_mat = get_simmat_from_metrics(metrics)
    sim_mat = sim_mat.unsqueeze(0) * 255
    writer.add_image(metric_name, sim_mat, it)


def average_q_px_dist_per_head_per_block(title, fname, loader, model):
    def get_features(name):
        def hook(model, input, output):
            qkvs[name] = output.detach()

        return hook

    for block_id, block in enumerate(model.blocks):
        block.attn.qkv.register_forward_hook(get_features(str(block_id)))

    patch_size = 16 if '16' in title.split('_')[2] else 32
    model.eval()
    qkvs = {}
    for batch_idx, (input, target) in enumerate(loader):
        _ = model(input)
        for block, qkv in qkvs.items():
            num_heads, scale = model.blocks[0].attn.num_heads, model.blocks[0].attn.scale
            B, N, CCC = qkv.shape
            C = CCC // 3
            qkv = qkv.reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            _, H, _, _ = attn.shape
            attn = attn.permute(1, 0, 2, 3)
            vect = torch.arange(N).reshape((1, N))
            dist_map = torch.sqrt(((vect - torch.transpose(vect, 0, 1)) % (N - 1) ** 0.5) ** 2 + (
                        (vect - torch.transpose(vect, 0, 1)) // (N - 1) ** 0.5) ** 2)
            per_head_dist_map = torch.sum(attn * torch.as_tensor(dist_map).to(device='cuda'), (1, 2, 3)) / torch.sum(
                attn, (1, 2, 3))
            qkvs[block] = per_head_dist_map * patch_size
        break
    vals = []
    for qkv in qkvs.values():
        vals.append(qkv.cpu().numpy())
    vals = np.asarray(vals)
    block_names = [str(i) for i in range(len(vals))]
    fig, ax = plt.subplots()
    for head in range(len(vals[0])):
        ax.scatter(block_names, vals[:, head], label='head_' + str(head))
    fig.suptitle(title)
    if len(vals[0]) < 7:
        ax.legend()
    ax.set_ylabel('Attention distance in Pixel')
    ax.set_xlabel('Block id')
    ax.grid(True, which='both')
    ax.set_ylim(ymax=180, ymin=0)
    plt.savefig(fname + '/Attn_dist.png')
    plt.close()


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


def get_CKA(val_path, model_t, model_t_name, model_c, model_c_name, data_loader):
    writer = SummaryWriter()
    modc_hooks = []
    for j, block in enumerate(model_c.blocks):
        tgt = f'blocks.{j}'
        hook = HookedCache(model_c, tgt)
        modc_hooks.append(hook)

    modt_hooks = []
    for j, block in enumerate(model_t.blocks):
        tgt = f'blocks.{j}'
        hook = HookedCache(model_t, tgt)
        modt_hooks.append(hook)

    metrics_ct = make_pairwise_metrics(modc_hooks, modt_hooks)
    metrics_tt = make_pairwise_metrics(modt_hooks, modt_hooks)

    with torch.no_grad():

        for it, (input, target) in enumerate(data_loader):
            # batch = batch.cuda()
            outv_c = model_c(input)
            outv_t = model_t(input)

            update_metrics(modc_hooks, modt_hooks, metrics_ct, "cka/ct", it, writer)
            update_metrics(modt_hooks, modt_hooks, metrics_tt, "cka/tt", it, writer)

            for hook0 in modc_hooks:
                for hook1 in modt_hooks:
                    hook0.clear()
                    hook1.clear()

    sim_mat = get_simmat_from_metrics(metrics_tt)
    plt.imshow(sim_mat)
    plt.title('Self CKA: ' + model_t_name)
    plt.savefig(val_path + '/Self_CKA.png')

    sim_mat = get_simmat_from_metrics(metrics_ct)
    plt.imshow(sim_mat)
    plt.title('CKA: ' + model_t_name + ' ' + model_c_name)
    plt.savefig(val_path + '/CKA' + model_t_name + '_' + model_c_name + '.png')
