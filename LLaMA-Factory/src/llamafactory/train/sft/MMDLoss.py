
# Reference:
# - PyTorch implementation reference: https://github.com/yiftachbeer/mmd_loss_pytorch/blob/master/mmd_loss.py

import torch
from torch import nn


class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY
    
def mmd_kl_divergence(stu: torch.Tensor, teacher: torch.Tensor, kernel=None):
    """
    Compute MMD loss between student and teacher.
    stu: [B1, L, V]
    teacher: [B2, L, V]
    return: scalar MMD loss
    """
    if kernel is None:
        kernel = RBF()

    mmd_loss = MMDLoss(kernel)

    # Average over batch dimension -> [L, V]
    stu_mean = stu.mean(dim=0)
    teacher_mean = teacher.mean(dim=0)

    # MMDLoss expects [N, D] form
    return mmd_loss(stu_mean, teacher_mean)