import random
import os
from pycocotools.coco import COCO

from yolox.data.datasets.coco import COCODataset, remove_useless_info
from yolox.data.datasets.datasets_wrapper import CacheDataset
from yolox.data.dataloading import get_yolox_datadir


# THS, Copied from yolox.data.dataset.coco


class Dataset(COCODataset):
    def __init__(
            self,
            data_dir=None,
            json_file="instances_train2017.json",
            name="train2017",
            img_size=(416, 416),
            preproc=None,
            cache=False,
            cache_type="ram",
            train_subset_fract: float = None,
            seed: int = 2024,
    ):
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "COCO")
        self.data_dir = data_dir
        self.json_file = json_file

        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        remove_useless_info(self.coco)
        self.ids = self.coco.getImgIds()

        # >>> MOD: subset sampling TMP TMP
        if train_subset_fract is not None:
            random.seed(seed)
            random.shuffle(self.ids)
            n = int(len(self.ids) * train_subset_fract)
            self.ids = self.ids[:n]
        # <<< MOD

        self.num_imgs = len(self.ids)
        self.class_ids = sorted(self.coco.getCatIds())
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in self.cats])
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.annotations = self._load_coco_annotations()

        path_filename = [os.path.join(name, anno[3]) for anno in self.annotations]
        CacheDataset.__init__(
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir=data_dir,
            cache_dir_name=f"cache_{name}",
            path_filename=path_filename,
            cache=cache,
            cache_type=cache_type
        )
