import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn

from .base_yolox import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()

        # ams_loss = AMSoftmaxLoss(embedding_dim=320, no_classes=num_classes, scale=10.0, reduction="none")
        # contrastive_loss = SupervisedContrastiveLoss()

        self.embedding_loss = None
        self.embedding_loss_weight = 1
        self.save_history_ckpt = False

        # --------------- transform config ----------------- #
        # prob of applying mosaic aug
        self.mosaic_prob = 1 # 0.2
        # prob of applying mixup aug
        self.mixup_prob = 1 # 0.2
        # prob of applying hsv aug
        self.hsv_prob = 1.0
        # prob of applying flip aug
        self.flip_prob = 0.5
        # rotation angle range, for example, if set to 2, the true range is (-2, 2)
        self.degrees = 10.0
        # translate range, for example, if set to 0.1, the true range is (-0.1, 0.1)
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2)
        # apply mixup aug or not
        self.enable_mixup = True
        self.mixup_scale = (0.5, 1.5)
        # shear angle range, for example, if set to 2, the true range is (-2, 2)
        self.shear = 2.0

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN  # , YOLOXHead # THS
        from .embedding_train_yolo_head import YOLOXHead

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
                             embedding_loss=self.embedding_loss, embedding_loss_weight=self.embedding_loss_weight)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model
