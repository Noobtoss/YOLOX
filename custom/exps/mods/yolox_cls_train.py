import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger

from .yolox_base import Exp as MyExp


# THS, based on: yolox.exp.yolox_base.py


class Exp(MyExp):
    def __init__(self):
        super().__init__()

        self.cls_emb_loss        = None   # SupervisedContrastiveLoss()
        self.cls_emb_weight      = None   # 1
        self.cls_dropout_p       = None   # 0.5
        self.cls_train_scheduler = None   # YoloHeadClsScheduler()
        self.save_history_ckpt   = False  # True

    def get_cls_train_scheduler(self):
        from .yolo_head_cls_scheduler import YoloHeadClsScheduler
        scheduler = YoloHeadClsScheduler(
            model=self.model,
            cls_emb_weight=self.cls_emb_weight,
            cls_dropout_p=self.cls_dropout_p,
            max_epoch=self.max_epoch,
        )
        return scheduler

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN  # , YOLOXHead # THS
        from .yolo_head_cls_train import YOLOXHead

        if self.cls_emb_loss is None:
            raise NotImplementedError("cls_emb_loss must be set before calling get_model().")
        else:
            print(f"cls_emb_loss: {self.cls_emb_loss}")

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act,
                             cls_emb_loss=self.cls_emb_loss,
                             cls_emb_weight=self.cls_emb_weight,
                             cls_dropout_p=self.cls_dropout_p)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model

    def get_trainer(self, args):
        from .trainer import Trainer
        trainer = Trainer(self, args)
        # NOTE: trainer shouldn't be an attribute of exp object
        return trainer
