import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

class BernoulliGammaLoss(nn.Module):
    def __init__(self, reduction="mean", eps=1e-3):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, y):
        """
        pi:    (B, ...) Bernoulli probability, in (0,1)
        alpha: (B, ...) Gamma shape > 0
        beta:  (B, ...) Gamma scale > 0
        y:     (B, ...) target values, either 0 or strictly > 0
        """
        pi = pred[:, 0]
        alpha = pred[:, 1]
        beta = pred[:, 2]

        if torch.any(y < - self.eps):
            warnings.warn(
                "Some target values are negative, check if any normalisation is applied"
                f"min={y.min()}, max={y.max()}", UserWarning)
        #  Runtime checks with warnings
        if torch.any((pi < 0) | (pi > 1)):
            warnings.warn(
                "Some pi values are outside (0,1). They will be clamped."
                f"min={pi.detach().min().item()}, max={pi.detach().max().item()}", UserWarning)
            pi = torch.clamp(pi, 1e-6, 1 - self.eps)
        if torch.any(alpha < 0):
            warnings.warn(
                "Some alpha values are <= 0. They will be clamped."
                f"min={alpha.detach().min().item()}, max={alpha.detach().max().item()}", UserWarning)
            alpha = torch.clamp(alpha, 0, None)
        if torch.any(beta < 1e-6):
            warnings.warn(
                f"Some beta values are <= {1e-6}. They will be clamped."
                f"min={beta.detach().min().item()}, max={beta.detach().max().item()}", UserWarning)
            beta = torch.clamp(beta, self.eps, None)

        #  Case y == 0
        loss_zero = -torch.log1p(-pi)
        loss_pos = (
            - torch.log(pi)
            + torch.lgamma(alpha)
            + alpha * torch.log(beta)
            - (alpha - 1) * torch.log(torch.clamp(y, min=self.eps))
            + y / beta
        )

        occurence_mask = (y > self.eps).float()
        loss = (1 - occurence_mask) * loss_zero + occurence_mask * loss_pos

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
