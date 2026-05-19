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
    def __init__(self, top_rel: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.top_rel = top_rel

    def _masking(self, metric):
        k = max(1, int(len(metric) * self.top_rel))
        thresh = metric.topk(k).values[-1]
        return metric >= thresh


class ConfMask(Masking, ConfWeight):
    def __call__(self, pred_scores, target_scores):
        conf = super().__call__(pred_scores, target_scores)
        return self._masking(conf)


class MaskFactory:
    @staticmethod
    def get(mask: str = None, **kwargs):
        if mask is None or mask == "None":
            return None
        elif mask == "conf":
            return ConfMask(**kwargs)
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
        loss = torch.zeros(1, device=cls_feats.device)
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
