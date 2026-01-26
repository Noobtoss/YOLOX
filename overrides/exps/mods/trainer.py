from loguru import logger

from yolox.core import Trainer as BaseTrainer


# THS, based on: yolox.core.trainer


class Trainer(BaseTrainer):
    def __init__(self, exp, args):
        super().__init__(exp, args)
        self.extra_scheduler = None

    def before_train(self):
        super().before_train()
        if hasattr(self.exp, 'get_extra_scheduler'):
            self.extra_scheduler = self.exp.get_extra_scheduler()

    def before_epoch(self):
        if self.extra_scheduler is not None:
            self.extra_scheduler(self.epoch + 1)
        super().before_epoch()
