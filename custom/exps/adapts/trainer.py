from loguru import logger

from yolox.core import Trainer as BaseTrainer


# THS, based on: yolox.core.trainer


class Trainer(BaseTrainer):
    def __init__(self, exp, args):
        super().__init__(exp, args)

    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))

        if self.model.head.cls_emb_weight is None:
            start, end = 2.0, 0.2
            cls_emb_weight = end + (start - end) * (1 - self.epoch / self.max_epoch)
            self.model.head.dynamic_cls_emb_weight = cls_emb_weight

        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            logger.info("--->No mosaic aug now!")
            self.train_loader.close_mosaic()
            logger.info("--->Add additional L1 loss now!")
            if self.is_distributed:
                self.model.module.head.use_l1 = True
            else:
                self.model.head.use_l1 = True
            self.exp.eval_interval = 1
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")
