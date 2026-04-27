import os
import random

from mods.yolox_bmvc_2026 import Exp as MyExp
from mods.sup_con_loss import SupConLoss


class Exp(MyExp):

    def __init__(self):
        super().__init__()
        self.num_classes = 15

        self.cls_emb_loss = SupConLoss()  # temperature=0.07
        self.cls_emb_weight = 0
        self.train_subset_fract = None
        self.train_min_cat_fract = None
        self.seed = random.randint(0, 2**32 - 1)

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.exp_name = f"{self.exp_name}_baseline"

        # ---------------- dataloader config ---------------- #

        # Define yourself dataset path
        self.data_dir = "datasets/Oktoberfest"
        self.train_ann = "annotation_train.json"
        self.val_ann = "annotation_test.json"

        # --------------  training config --------------------- #

        self.max_epoch = 100
        self.data_num_workers = 4
        self.eval_interval = 1

        # ---------------- semmel config ---------------- #
        self.num_classes = 15
        self.names = {
            1: 'Bier',
            2: 'Biermass',
            3: 'Weissbier',
            4: 'Cola',
            5: 'Wasser',
            6: 'Currywurst',
            7: 'Weisswein',
            8: 'Apfelschorle',
            9: 'Jaegermeister',
            10: 'Pommes',
            11: 'Burger',
            12: 'Williamsbirne',
            13: 'Almbrezel',
            14: 'Brotzeitkorb',
            15: 'Kaesespaetzle',
        }
        self.img_size = (1280, 1280) # (640, 640)  # (height, width)

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
