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

        # ams_loss = AMSoftmaxLoss(embedding_dim=320, no_classes=num_classes, scale=10.0, reduction="none")
        # contrastive_loss = SupervisedContrastiveLoss()

        self.embedding_loss = None
        self.embedding_weight = None
        self.save_history_ckpt = True

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN  # , YOLOXHead # THS
        from .yolo_head_embedding_train import YOLOXHead

        if self.embedding_loss is None:
            raise NotImplementedError("embedding_loss must be set before calling get_model().")
        else:
            print(f"embedding_loss: {self.embedding_loss}")

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act,
                             embedding_loss=self.embedding_loss, embedding_weight=self.embedding_weight)
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
