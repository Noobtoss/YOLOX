import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses, reducers


class UnpackReducer(reducers.BaseReducer):
    def element_reduction(self, losses, loss_indices, embeddings, labels):
        sorted_indices = torch.argsort(loss_indices)
        return losses[sorted_indices]


class NormalizeFeats(nn.Module):
    """Wraps any embedding loss and L2-normalizes embeddings before forwarding."""

    def __init__(self, loss: nn.Module):
        super().__init__()
        self.loss = loss

    def forward(self, feats, *args, **kwargs):
        return self.loss(F.normalize(feats, dim=1), *args, **kwargs)


class FeatLossFactory:
    @staticmethod
    def get(loss: str = None, **kwargs):
        if loss is None or loss == "None":
            return None
        elif loss == "sup_con_loss":
            # https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#supconloss
            kwargs = {k: v for k, v in kwargs.items() if k in inspect.signature(losses.SupConLoss).parameters}
            return NormalizeFeats(losses.SupConLoss(**kwargs, reducer=UnpackReducer()))
        elif loss == "general_lifted_struct":
            # https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#generalizedliftedstructureloss
            kwargs = {k: v for k, v in kwargs.items() if
                      k in inspect.signature(losses.GeneralizedLiftedStructureLoss).parameters}
            return NormalizeFeats(losses.GeneralizedLiftedStructureLoss(**kwargs, reducer=UnpackReducer()))
        else:
            raise ValueError(f"Unknown feat loss type: '{loss}'")


class ConfWeight:
    def __init__(self, **kwargs):
        pass

    def __call__(self, pred_scores, target_scores):
        return pred_scores.sigmoid().max(-1).values


class WeightFactory:
    @staticmethod
    def get(weight: str = None, **kwargs):
        if weight is None or weight == "None":
            return None
        elif weight == "conf":
            return ConfWeight()
        else:
            raise ValueError(f"Unknown weight type: '{weight}'")


class Masking:
    def __init__(self, mask_pct: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.mask_pct = mask_pct

    def _masking(self, metric):
        k = max(1, int(len(metric) * (1 - self.mask_pct)))
        thresh = metric.topk(k).values[-1]
        return metric >= thresh


class ConfMask(Masking, ConfWeight):
    def __call__(self, pred_scores, target_scores):
        conf = super().__call__(pred_scores, target_scores)
        return self._masking(conf)


class RandMask:
    def __init__(self, mask_pct: float = 0.4, **kwargs):
        self.mask_pct = mask_pct

    def __call__(self, pred_scores, target_scores):
        k = max(1, int(len(pred_scores) * self.mask_pct))
        mask = torch.zeros(len(pred_scores), dtype=torch.bool)
        indices = torch.randperm(len(pred_scores))[:k]
        mask[indices] = True
        return ~mask


class RandMaskBalanced:
    def __init__(self, mask_pct: float = 0.4, min_per_class: int = 4, **kwargs):
        self.mask_pct = mask_pct
        self.min_per_class = min_per_class

    def __call__(self, pred_scores, target_scores):
        target_cls = target_scores.max(-1).indices
        n = len(pred_scores)
        k = max(1, int(n * self.mask_pct))
        mask = torch.zeros(n, dtype=torch.bool)

        # Guarantee at least min_per_class per unique class
        for cls in target_cls.unique():
            cls_indices = (target_cls == cls).nonzero(as_tuple=True)[0]
            k_cls = min(self.min_per_class, len(cls_indices))
            chosen = cls_indices[torch.randperm(len(cls_indices))[:k_cls]]
            mask[chosen] = True

        # Fill remaining budget with random unchosen indices
        remaining = k - mask.sum().item()
        if remaining > 0:
            unmasked = (~mask).nonzero(as_tuple=True)[0]
            extra = unmasked[torch.randperm(len(unmasked))[:remaining]]
            mask[extra] = True

        return ~mask


class MaskFactory:
    @staticmethod
    def get(mask: str = None, **kwargs):
        if mask is None or mask == "None":
            return None
        elif mask == "conf":
            return ConfMask(**kwargs)
        elif mask == "rand":
            return RandMask(**kwargs)
        elif mask == "rand_balanced":
            return RandMaskBalanced(**kwargs)
        else:
            raise ValueError(f"Unknown mask type: '{mask}'")

class ClsFeatLoss(nn.Module):
    def __init__(self, loss: str, mask: str = None, weight: str = None, **kwargs):
        super().__init__()
        self.loss = FeatLossFactory.get(loss, **kwargs)
        self.mask = MaskFactory.get(mask, **kwargs)
        self.weight = WeightFactory.get(weight, **kwargs)

    def forward(
            self,
            cls_feats: torch.Tensor,
            pred_scores: torch.Tensor,
            target_scores: torch.Tensor
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=cls_feats.device)
        target_cls = target_scores.max(-1).indices
        if self.mask is not None:
            mask = self.mask(cls_feats, target_scores)
            if not mask.sum():
                return loss
            cls_feats = cls_feats[mask]
            target_cls = target_cls[mask]
            pred_scores = pred_scores[mask]
            target_scores = target_scores[mask]

        loss_per_element = self.loss(cls_feats, target_cls)

        if self.weight is not None:
            weight = self.weight(pred_scores, target_scores)
            weight = weight / weight.sum()
            loss += (loss_per_element * weight).sum()
        else:
            loss += loss_per_element.mean()

        return loss
