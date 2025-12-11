import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log


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
    y_safe = torch.where(y == 0, torch.ones_like(y), y)  # prevent div by 0
    result = x / y_safe
    result = torch.where(y == 0, torch.zeros_like(result), result)
    return result


class TargetedSupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, target_ids=None):
        """
        Implementation of the loss described in the paper Supervised Contrastive Learning :
        https://arxiv.org/abs/2004.11362

        :param temperature: int
        """
        super().__init__()
        self.temperature = temperature
        if target_ids is None:
            raise ValueError("target_ids must be set for targeted sampling.")
        self.target_ids = torch.tensor(target_ids)

    def targeted_sampling(self, projections, onehot_targets):
        targets = onehot_targets.argmax(dim=1)

        if self.target_ids.device != targets.device:
            self.target_ids = self.target_ids.to(targets.device)
        mask = torch.isin(targets, self.target_ids)
        projections = projections[mask]
        targets = targets[mask]

        projections = F.normalize(projections, dim=1)

        return projections, targets

    def forward(self, projections, targets):
        """
        :param projections: torch.Tensor, shape [batch_size, projection_dim]
        :param targets: torch.Tensor, shape [batch_size]
        :return: torch.Tensor, scalar
        """
        projections, targets = self.targeted_sampling(projections, targets)

        if targets.shape[0] == 0:
            return torch.tensor(0.0, device=targets.device, dtype=targets.dtype)

        device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")

        dot_product_tempered = torch.mm(projections, projections.T) / self.temperature
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
                torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )

        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)

        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        supervised_contrastive_loss_per_sample = divide_no_nan(
            torch.sum(log_prob * mask_combined, dim=1), cardinality_per_samples
        )

        return supervised_contrastive_loss_per_sample
