import torch
from torch.nn.modules import Module
from torch.nn import functional as F
from torch import Tensor
import numpy as np


def probability_from_long_distance_relation(attn):
    def distance2proba(per_head_avg_distance, N):
        proba = per_head_avg_distance/np.sqrt(N)
        proba = .5*torch.exp(5*proba)
        return proba.cpu().detach().numpy()

    B, H, N, _ = attn.shape
    attn = attn.permute(1, 0, 2, 3)
    dist_map = np.zeros((H, B, N, N))
    for i in range(N):
        for j in range(N):
            dist_map[:, :, i, j] = np.sqrt(((j-i)%np.sqrt(N))**2 + ((j-i)//np.sqrt(N))**2)
    per_head_probability = distance2proba(torch.mean(attn * torch.as_tensor(dist_map, dtype=torch.float16).to(device='cuda'), (1, 2, 3)), N)
    return per_head_probability


class _DropoutNd(Module):
    __constants__ = ['p', 'inplace']
    p: float
    inplace: bool

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super(_DropoutNd, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def extra_repr(self) -> str:
        return 'p={}, inplace={}'.format(self.p, self.inplace)


class CustomDropout(_DropoutNd):
    def forward(self, input: Tensor, attn: Tensor) -> Tensor:
        per_head_p = probability_from_long_distance_relation(attn)
        B, N, C = input.shape
        B, H, N, _ = attn.shape
        input = input.reshape(B, N, H, C // H).permute(2, 0, 1, 3)
        for i in range(H):
            input[i] = F.dropout(input[i], per_head_p[i], self.training, self.inplace)
        return input.permute(1, 2, 0, 3).reshape(B, N, C)