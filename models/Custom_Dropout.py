import torch.nn.modules.dropout
from torch.nn import functional as F
from torch import Tensor

class CustomDropout(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor:
        p = probability_from_long_ditance_relation(input)
        return F.dropout(input, p, self.training, self.inplace)