from yolox.models import YOLOX


class Scheduler:
    def __init__(self, model: YOLOX, cls_emb_weight: float = 1, cls_dropout_p: float = 0.5, max_epoch: int = None):
        self.model = model
        self.cls_emb_weight = cls_emb_weight
        self.cls_dropout_p = cls_dropout_p
        self.max_epoch = max_epoch

    def __call__(self, epoch: int):
        pass
