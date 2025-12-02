import os

from adapts.yolox_cls_train import Exp as MyExp
from adapts.ams_loss import AMSoftmaxLoss
from adapts.sup_contrastive_loss import SupervisedContrastiveLoss
from adapts.targeted_sup_contrastive_loss import TargetedSupervisedContrastiveLoss
from adapts.yolo_head_cls_scheduler import YoloHeadClsScheduler


class Exp(MyExp):

    def __init__(self):
        super().__init__()

        # ams_loss = AMSoftmaxLoss(cls_emb_dim=320, no_classes=37, scale=10.0, reduction="none")
        sup_contrastive_loss = SupervisedContrastiveLoss()  # temperature=0.07
        # self.target_ids = [24, 34]  # [25, 31, 35]  # [24, 34, 25, 31, 35]
        # targeted_sup_contrastive_loss = TargetedSupervisedContrastiveLoss(temperature=0.07, target_ids=self.target_ids)

        # self.cls_emb_loss = ams_loss
        self.cls_emb_loss = sup_contrastive_loss
        self.cls_emb_weight = 0  # 1
        self.cls_dropout_p = None  # 0.5
        self.cls_train_scheduler = YoloHeadClsScheduler

        # prob of applying mosaic aug
        self.mosaic_prob = 1
        # prob of applying mixup aug
        self.mixup_prob = 1

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.exp_name = f"{self.exp_name}_baseline"

        # ---------------- dataloader config ---------------- #

        # Define yourself dataset path
        self.data_dir = "datasets/Images05"
        self.train_ann = "annotation_train.json"
        self.val_ann = "annotation_test.json"

        # --------------  training config --------------------- #

        self.max_epoch = 100
        self.data_num_workers = 4
        self.eval_interval = 1

        # ---------------- semmel config ---------------- #

        self.num_classes = 37
        self.names = {
            1: "Backware",
            2: "Bauernbrot",
            3: "Floesserbrot",
            4: "Salzstange",
            5: "Sonnenblumensemmel",
            6: "Kuerbiskernsemmel",
            7: "Roggensemmel",
            8: "Dinkelsemmel",
            9: "LaugenstangeSchinkenKaese",
            10: "Pfefferlaugenbrezel",
            11: "KernigeStange",
            12: "Schokocroissant",
            13: "Apfeltasche",
            14: "Quarktasche",
            15: "Mohnschnecke",
            16: "Nussschnecke",
            17: "Vanillehoernchen",
            18: "Osterei",
            19: "Osterbrezel",
            20: "Kirschtasche",
            21: "Fruechteschiffchen",
            22: "Anisbrezel",
            23: "Doppelsemmel",
            24: "Fruestuecksemmel",
            25: "Kaisersemmel",
            26: "Kornknacker",
            27: "Landbrot",
            28: "Laugenbrezel",
            29: "Laugenstange",
            30: "Laugenzopf",
            31: "Mohnsemmel",
            32: "Mohnstange",
            33: "Partybrot",
            34: "Sandwichbroetchen",
            35: "Sesamsemmel",
            36: "Sesamstange",
            37: "Vollgutsemmel"
        }
        self.img_size = (1280, 1280)  # (640, 640)  # (height, width)

        # ---------------- model config ---------------- #

        scale = "yolox_x"  # "yolox_m" # "yolox_l" # "yolox_x"

        if scale == "yolox_s":
            self.depth = 0.33
            self.width = 0.50
        if scale == "yolox_m":
            self.depth = 0.67
            self.width = 0.75
        if scale == "yolox_l":
            self.depth = 1.0
            self.width = 1.0
        if scale == "yolox_x":
            self.depth = 1.33
            self.width = 1.25

        # self.ckpt = f"checkpoints/{scale}.pth"
