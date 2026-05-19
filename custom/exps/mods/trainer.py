import warnings
import time
import torch
from yolox.core import Trainer as _Trainer


# THS, Copied from yolox.core.trainer


class Trainer(_Trainer):
    def __init__(self, exp, args):
        warnings.warn("[Modded] Trainer")
        super().__init__(exp, args)
        self.extra_scheduler = None

    def before_train(self):
        super().before_train()
        # or is lr_scheduler default
        self.cls_feat_proj_head_lr_scheduler = self.exp.get_lr_scheduler(
            getattr(self.exp, "cls_feat_proj_head_lr", None) or self.exp.basic_lr_per_img * self.args.batch_size,
            self.max_iter
        )
        if hasattr(self.exp, 'get_extra_scheduler'):
            self.extra_scheduler = self.exp.get_extra_scheduler()

    def before_epoch(self):
        if self.extra_scheduler is not None:
            self.extra_scheduler(self.epoch + 1)
        super().before_epoch()

    def train_one_iter(self):
        iter_start_time = time.time()

        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            outputs = self.model(inps, targets)

        loss = outputs["total_loss"]

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)
        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        # >>> MOD
        cls_feat_proj_head_lr = self.cls_feat_proj_head_lr_scheduler.update_lr(self.progress_in_iter + 1)
        self.optimizer.param_groups[-1]["lr"] = cls_feat_proj_head_lr

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            cls_feat_proj_head_lr=cls_feat_proj_head_lr,
            **outputs,
        )
        # <<< MOD

    def after_iter(self):
        super().after_iter()
        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger.add_scalar(
                    "train/cls_feat_proj_head_lr", self.meter["cls_feat_proj_head_lr"].latest, self.progress_in_iter)
            if self.args.logger == "wandb":
                metrics = {"train/cls_feat_proj_head_lr": self.meter["cls_feat_proj_head_lr"].latest}
                self.wandb_logger.log_metrics(metrics, step=self.progress_in_iter)
            if self.args.logger == 'mlflow':
                logs = {"train/cls_feat_proj_head_lr": self.meter["cls_feat_proj_head_lr"].latest}
                self.mlflow_logger.on_log(self.args, self.exp, self.epoch + 1, logs)
