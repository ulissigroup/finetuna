import torch
import torch.nn as nn


class RelativeL2MAELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum"]

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        diff = input - target
        relative = (target**2).sum(axis=1).sqrt()
        relative_diff = (diff.T / relative).T
        dists = torch.norm(relative_diff, p=2, dim=-1)
        if self.reduction == "mean":
            return torch.mean(dists)
        elif self.reduction == "sum":
            return torch.sum(dists)
