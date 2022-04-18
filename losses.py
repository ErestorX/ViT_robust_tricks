import torch


class AttentionProfileLoss(torch.nn.Module):
    def __init__(self, policy='max', operation='mean'):
        super(AttentionProfileLoss, self).__init__()
        self.policy = policy
        self.operation = operation

    def forward(self, x):
        if self.operation == 'mean':
            x = torch.mean(x)
        elif self.operation == 'std':
            x = torch.std(x)
        if self.policy == 'max':
            return torch.sum(1 - x)
        elif self.policy == 'min':
            return torch.sum(x)