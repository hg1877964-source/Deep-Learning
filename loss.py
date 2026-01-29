import torch
import torch.nn as nn

class ScaleInvariantLoss(nn.Module):
    """
    实现尺度不变损失函数，消除不同场景间绝对深度尺度的影响
    """
    def __init__(self, lam=0.5):
        super().__init__()
        self.lam = lam

    def forward(self, pred, target):
        mask = target > 0
        if not mask.any(): return torch.tensor(0.0)

        # 对数空间计算
        diff = torch.log(pred[mask]) - torch.log(target[mask])
        # 公式: sqrt(mean(d^2) - lambda * mean(d)^2)
        loss = torch.sqrt((diff**2).mean() - self.lam * (diff.mean()**2))
        return loss