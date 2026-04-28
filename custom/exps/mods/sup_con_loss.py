import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/google-research/google-research/blob/master/supcon/losses.py#L99
# https://github.com/HobbitLong/SupContrast/blob/master/losses.py
# https://github.com/GuillaumeErhard/Supervised_contrastive_loss_pytorch/blob/main/loss/spc.py
# https://github.com/huggingface/sentence-transformers/tree/master/sentence_transformers/losses


def divide_no_nan(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    PyTorch equivalent of tf.math.divide_no_nan.
    Returns x / y, with zeros where y == 0.
    Differentiable and safe for autograd.
    """
    y_safe = torch.where(y == 0, torch.ones_like(y), y)
    result = x / y_safe
    result = torch.where(y == 0, torch.zeros_like(result), result)
    return result


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, projections: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        projections = F.normalize(projections, dim=1)   # L2 normalizes

        sim = torch.mm(projections, projections.T) / self.temperature
        sim = sim - sim.max(dim=1, keepdim=True)[0]  # numerical stability
        exp_sim = torch.exp(sim)

        mask_self = (1 - torch.eye(projections.shape[0], device=projections.device)).bool()
        mask_pos = (targets.unsqueeze(1) == targets.unsqueeze(0)) & mask_self

        # log prob for each pair
        denom = exp_sim.masked_fill(~mask_self, 0).sum(dim=1, keepdim=True)
        log_prob = sim - torch.log(denom + 1e-8)

        loss_per_sample = divide_no_nan(
            (log_prob * mask_pos).sum(dim=1),
            mask_pos.sum(dim=1).float()
        )

        return loss_per_sample
