import math


class ClsFeatScheduler:
    def __init__(self, name, cls_feat, total_epochs, **kwargs):
        self.name = name
        self.cls_feat = cls_feat
        self.total_epochs = total_epochs

    def update_cls_feat(self, epoch):
        if self.name == "anti_cos_decay":
            # Warm-up: grows from 0 to cls_feat
            return self.cls_feat - cos_lr(self.cls_feat, self.total_epochs, epoch)
        elif self.name == "constant":
            return self.cls_feat


def cos_lr(lr, total_iters, iters):
    """Cosine learning rate"""
    lr *= 0.5 * (1.0 + math.cos(math.pi * iters / total_iters))
    return lr
