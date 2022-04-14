import torch


class AttentionProfileLoss(torch.nn.Module):
    def __init__(self, policy='global'):
        super(AttentionProfileLoss, self).__init__()
        self.policy = policy

    def forward(self, x):
        x = torch.mean(x.swapaxes(0, 1), dim=(1, 2))
        if self.policy == 'global':
            return torch.sum(1 - x)
        elif self.policy == 'local':
            return torch.sum(x)