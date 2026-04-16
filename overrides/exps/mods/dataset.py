import os
import random
from collections import defaultdict

from pycocotools.coco import COCO

from yolox.data.dataloading import get_yolox_datadir
from yolox.data.datasets.coco import COCODataset, remove_useless_info
from yolox.data.datasets.datasets_wrapper import CacheDataset


# THS, Copied from yolox.data.dataset.coco


def coco_subsample(coco: COCO, subset_fract: float, min_cat_fract: float = None, seed: int = 2024) -> COCO:
    random.seed(seed)
    all_ids = set(coco.getImgIds())
    selected = set()
    total_budget = int(len(all_ids) * subset_fract)
    phase_01_budget_fract = 0.1
    phase_02_budget_fract = 0.7

    cat_ids = coco.getCatIds()

    # phase 1: equal distribution sampling
    budget = int(phase_01_budget_fract * total_budget)
    for cat_id in cat_ids:
        per_cat_budget = max(1, int(budget / len(cat_ids)))
        cat_imgs = coco.getImgIds(catIds=[cat_id])
        available = list(set(cat_imgs) - selected)
        random.shuffle(available)
        for img_id in available:
            if per_cat_budget <= 0:
                break
            selected.add(img_id)
            per_cat_budget -= 1

    # phase 2: guarantee minimum per category, rarest first
    total_anns = len(coco.getAnnIds())
    cat_ann_counts = {c: len(coco.getAnnIds(catIds=[c])) for c in cat_ids}
    cat_ids.sort(key=lambda c: cat_ann_counts[c])  # rarest first
    min_cat_fract = min_cat_fract or cat_ann_counts[cat_ids[0]] / total_anns
    min_cat_fract = min(min_cat_fract, 1.0 / len(cat_ids))
    per_cat_ann_min = {c: max(1, int(min_cat_fract * total_anns)) for c in cat_ids}

    img_cat_ann_count = defaultdict(lambda: defaultdict(int))
    for ann in coco.loadAnns(coco.getAnnIds()):
        img_cat_ann_count[ann["image_id"]][ann["category_id"]] += 1

    budget = int(phase_02_budget_fract * (total_budget - len(selected)))
    for cat_id in cat_ids:
        cat_imgs = coco.getImgIds(catIds=[cat_id])
        available = list(set(cat_imgs) - selected)
        random.shuffle(available)
        for img_id in available:
            if budget <= 0:
                break
            if per_cat_ann_min[cat_id] <= 0:
                break
            selected.add(img_id)
            for c, count in img_cat_ann_count[img_id].items():
                per_cat_ann_min[c] -= count
            budget -= 1

    # phase 3: random fill up to budget (or remaining after overflow)
    budget = int(total_budget - len(selected))
    remaining = list(all_ids - selected)
    random.shuffle(remaining)
    selected.update(remaining[:max(0, budget)])

    # build new COCO object
    new_coco = COCO()
    new_coco.dataset["images"] = [img for img in coco.dataset["images"] if img["id"] in selected]
    new_coco.dataset["annotations"] = [ann for ann in coco.dataset["annotations"] if ann["image_id"] in selected]
    new_coco.dataset["categories"] = coco.dataset["categories"]
    new_coco.createIndex()
    return new_coco


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
            train_min_cat_fract: float = None,
            seed: int = 2024,
    ):
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "COCO")
        self.data_dir = data_dir
        self.json_file = json_file

        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        # >>> MOD
        if train_subset_fract is not None:
            self.coco = coco_subsample(self.coco, train_subset_fract, train_min_cat_fract, seed)
        # <<< MOD
        remove_useless_info(self.coco)
        self.ids = self.coco.getImgIds()
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
            self,
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir=data_dir,
            cache_dir_name=f"cache_{name}",
            path_filename=path_filename,
            cache=cache,
            cache_type=cache_type
        )
