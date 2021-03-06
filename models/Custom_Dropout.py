import torch
from torch.nn.modules import Module
from torch.nn import functional as F
from torch import Tensor


def probability_from_long_distance_relation(attn, mode='exp', bin_do_val=0.15, threshold=4, layer=10, exp_mul=0.25):
    def distance2proba(per_head_avg_distance, N):
        if mode == 'exp':
            proba = per_head_avg_distance/(N ** 0.5)
            proba = .5/torch.exp(exp_mul * layer * proba)
        elif mode == 'negexp':
            proba = per_head_avg_distance / (N ** 0.5)
            proba = .5 - .5 / torch.exp(exp_mul * layer * proba)
        elif mode == 'square':
            proba = per_head_avg_distance / (N ** 0.5)
            proba[proba <= threshold] = 0
            proba[proba > threshold] = bin_do_val
        else:
            proba = per_head_avg_distance * 0.0
        return proba.cpu().detach().numpy()


    B, H, N, _ = attn.shape
    attn = attn.permute(1, 0, 2, 3)
    N = N - 1
    attn = attn[:, :, 1:, 1:]
    id_dist = torch.arange(N).reshape((1, N)) - torch.arange(N).reshape((N, 1))
    dist_map = torch.sqrt((torch.abs(id_dist) % N ** 0.5) ** 2 + (id_dist // N ** 0.5) ** 2)
    per_head_dist = torch.sum(attn * dist_map.to(device='cuda'), (1, 2, 3)) / torch.sum(attn, (1, 2, 3))
    return distance2proba(per_head_dist, N)


class _DropoutNd(Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False) -> None:
        super(_DropoutNd, self).__init__()
        self.inplace = inplace

    def extra_repr(self) -> str:
        return 'inplace={}'.format(self.inplace)


class CustomDropout(_DropoutNd):
    def __init__(self, mode='exp', threshold=4, layer=10, exp_mul=0.25):
        super().__init__()
        self.threshold = threshold
        self.mode = mode
        self.layer = layer
        self.exp_mul = exp_mul

    def forward(self, input: Tensor, attn: Tensor) -> Tensor:
        per_head_p = probability_from_long_distance_relation(attn, mode=self.mode, threshold=self.threshold, layer=self.layer, exp_mul=self.exp_mul)
        B, N, C = input.shape
        B, H, N, _ = attn.shape
        input = input.reshape(B, N, H, C // H).permute(2, 0, 1, 3)
        for i in range(H):
            input[i] = F.dropout(input[i], per_head_p[i], self.training, self.inplace)
        return input.permute(1, 2, 0, 3).reshape(B, N, C)