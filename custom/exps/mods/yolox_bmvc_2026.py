import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger

from .yolox_base import Exp as MyExp


# THS, Copied from yolox.exp.yolox_base.py


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.save_history_ckpt   = False  # True
        self.cls_feat_loss       = None   # SupervisedContrastiveLoss()
        self.cls_feat_weight     = None   # 1
        self.save_history_ckpt   = False  # True
        self.train_subset_fract  = None
        self.train_min_cat_fract = None
        self.seed                = 2024

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN  # , YOLOXHead # THS
        from .yolo_head_bmvc_2026 import YOLOXHead

        if self.cls_feat_loss is None:
            raise NotImplementedError("cls_emb_loss must be set before calling get_model().")
        else:
            print(f"cls_emb_loss: {self.cls_feat_loss}")

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
            head = YOLOXHead(
                self.num_classes, self.width, in_channels=in_channels, act=self.act, cls_feat_loss=self.cls_feat_loss,
                cls_feat_weight=float(self.cls_feat_weight) if self.cls_feat_weight is not None else None,
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        """
        Get dataset according to cache and cache_type parameters.
        Args:
            cache (bool): Whether to cache imgs to ram or disk.
            cache_type (str, optional): Defaults to "ram".
                "ram" : Caching imgs to ram for fast training.
                "disk": Caching imgs to disk for fast training.
        """
        from yolox.data import COCODataset, TrainTransform
        from .dataset import Dataset

        return Dataset(
            name="Images",  # self.train_ann.split("annotation_")[-1].removesuffix(".json"),
            data_dir=self.data_dir,
            json_file=self.train_ann,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            cache=cache,
            cache_type=cache_type,
            train_subset_fract=float(self.train_subset_fract) if self.train_subset_fract is not None else None,
            train_min_cat_fract=float(self.train_min_cat_fract) if self.train_min_cat_fract is not None else None,
            seed=int(self.seed) if self.seed is not None else None,
        )
