import torch
from torch.nn.modules import Module
from torch.nn import functional as F
from torch import Tensor


def probability_from_long_distance_relation(attn):
    def distance2proba(per_head_avg_distance, N):
        proba = per_head_avg_distance/(N**0.5)
        proba = .5/torch.exp(5*proba)
        return proba.cpu().detach().numpy()

    B, H, N, _ = attn.shape
    attn = attn.permute(1, 0, 2, 3)
    vect = torch.arange(N).reshape((1, N))
    dist_map = torch.sqrt(((vect - torch.transpose(vect, 0, 1)) % N**0.5) ** 2 + ((vect - torch.transpose(vect, 0, 1)) // N**0.5) ** 2)
    per_head_dist = torch.sum(attn * torch.as_tensor(dist_map).to(device='cuda'), (1, 2, 3)) / torch.sum(attn, (1, 2, 3))
    return distance2proba(per_head_dist, N)


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