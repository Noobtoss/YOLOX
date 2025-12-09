from loguru import logger

from yolox.core import Trainer as BaseTrainer


# THS, based on: yolox.core.trainer


class Trainer(BaseTrainer):
    def __init__(self, exp, args):
        super().__init__(exp, args)
        self.cls_train_scheduler = None

    def before_train(self):
        super().before_train()
        if hasattr(self.exp, 'get_cls_train_scheduler'):
            self.cls_train_scheduler = self.exp.get_cls_train_scheduler()

    def before_epoch(self):
        if self.cls_train_scheduler is not None:
            self.cls_train_scheduler(self.epoch + 1)
        super().before_epoch()
