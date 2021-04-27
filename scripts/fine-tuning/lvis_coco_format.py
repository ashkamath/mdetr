# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
"""Preprocess the dataset in a format adequate to train MDETR.

Specifically, if an image I contains positive classes (c_1, c_2, ..., c_n), then we create the annotations
[(I, c_1), ... (I, c_n)]. Each annotation contains all the instances of the corresponding class.

To fully train MDETR as a detector, we also need to add some negative classes. We use LVIS annotations to retrieve
the negative classes for image I. Similarly, we create an annotation for each negative class.

The script also allows to sample a random subset of LVIS. In this case, we try to balance the subset, so that even the
1% subset contains at least one positive and one negative example for each class.
"""
import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser("Conversion script")

    parser.add_argument(
        "--data_path",
        default="",
        type=str,
        help="Path to the lvis v1 dataset",
    )

    parser.add_argument(
        "--subset_fract",
        default=1.0,
        type=float,
        help="Which fraction of the dataset to use. Leave as 1 to use the full dataset",
    )

    parser.add_argument(
        "--out_path",
        default="",
        type=str,
        help="Path where to export the resulting dataset.",
    )
    return parser.parse_args()


def clean_name(name):
    """Convert the LVIS class name into a query text for MDETR"""
    name = re.sub(r"\(.*\)", "", name)
    name = re.sub(r"_", " ", name)
    name = re.sub(r"  ", " ", name)
    return name


def get_subset(data, fract: float, imid2data):
    """Get a balanced subset containing the target fraction of the full dataset"""
    if fract == 1:
        return {i["id"] for i in data["images"]}

    random.seed(42)

    cat2img = defaultdict(set)
    img2cat = defaultdict(set)
    negcat2img = defaultdict(set)
    for ann in data["annotations"]:
        if ann["category_id"] in imid2data[ann["image_id"]]["not_exhaustive_category_ids"]:
            continue
        cat2img[ann["category_id"]].add(ann["image_id"])
        img2cat[ann["image_id"]].add(ann["category_id"])

    # Some classes are never exhaustively annotated. We pick some representants anyway
    not_exhaustive = set([c["id"] for c in data["categories"]]) - set(cat2img.keys())

    for ann in data["annotations"]:
        if ann["category_id"] not in not_exhaustive:
            continue
        cat2img[ann["category_id"]].add(ann["image_id"])
        img2cat[ann["image_id"]].add(ann["category_id"])

    for img in data["images"]:
        for cat in img["neg_category_ids"]:
            negcat2img[cat].add(img["id"])

    cat_count = Counter()
    neg_cat_count = Counter()
    for _, c in sorted([(cat["image_count"], cat["id"]) for cat in data["categories"]], reverse=True):
        cat_count[c] = 0
        neg_cat_count[c] = 0
    target_count = int(len(data["images"]) * fract)

    choosen_ids = set()
    exhausted_pos, exhausted_neg = set(), set()
    print(f"Selecting subset of size {target_count}...")
    for j in tqdm(range(target_count)):
        do_pos = j % 10 != 0
        current_counter = cat_count if do_pos else neg_cat_count
        exhausted_set = exhausted_pos if do_pos else exhausted_neg

        possible_img_set = set()
        for i in range(len(current_counter)):
            target_cat, _ = current_counter.most_common()[-(i + 1)]  # ith least common
            if target_cat in exhausted_set:
                continue
            if do_pos:
                possible_img_set = cat2img[target_cat] - choosen_ids
            else:
                possible_img_set = negcat2img[target_cat] - choosen_ids
            if len(possible_img_set) > 0:
                break
            else:
                exhausted_set.add(target_cat)

        assert len(possible_img_set) > 0
        possible_imgs = sorted(list(possible_img_set))
        cur_img = random.choice(possible_imgs)
        for c in img2cat[cur_img]:
            cat_count[c] += 1
        choosen_ids.add(cur_img)

        for neg in imid2data[cur_img]["neg_category_ids"]:
            neg_cat_count[neg] += 1
    assert cat_count.most_common()[-1][1] > 0, "Unable to find at least one image for each category"
    return choosen_ids


def convert(split, data_path, output_path, fract):

    print("Loading annotations...")
    with open(data_path / f"lvis_v1_{split}.json", "r") as f:
        data = json.load(f)

    imid2data = {x["id"]: x for x in data["images"]}
    kept_ids = get_subset(data, fract if split == "train" else 1, imid2data)

    img2ann = defaultdict(list)
    for datapoint in data["annotations"]:
        if datapoint["image_id"] in kept_ids:
            img2ann[datapoint["image_id"]].append(datapoint)

    print(f"Final size: {len(kept_ids)}")

    print(f"Dumping {split}...")
    next_img_id = 0
    next_id = 0

    categories = [{"supercategory": "object", "id": 1, "name": "object"}]
    annotations = []
    images = []
    id2catname = {c["id"]: clean_name(c["name"]) for c in data["categories"]}
    d_name = "lvis"

    for image_id in tqdm(kept_ids):
        cat2ann = defaultdict(list)
        for ann in img2ann[image_id]:
            cat2ann[ann["category_id"]].append(ann)

        filename = "/".join(imid2data[int(image_id)]["coco_url"].split("/")[-2:])
        for cat, ann_list in cat2ann.items():
            phrase = id2catname[cat]

            cur_img = {
                "file_name": filename,
                "height": imid2data[int(image_id)]["height"],
                "width": imid2data[int(image_id)]["width"],
                "id": next_img_id,
                "original_id": image_id,
                "coco_url": imid2data[int(image_id)]["coco_url"],
                "caption": phrase,
                "tokens_negative": [(0, len(phrase))],
                "dataset_name": d_name,
            }

            for ann in ann_list:
                cur_obj = {
                    "area": ann["area"],
                    "iscrowd": 0,
                    "category_id": 1,
                    "bbox": ann["bbox"],
                    "tokens_positive": [(0, len(phrase))],
                    "image_id": next_img_id,
                    "id": next_id,
                }

                next_id += 1
                annotations.append(cur_obj)

            next_img_id += 1
            images.append(cur_img)

        for cat in imid2data[image_id]["neg_category_ids"]:
            phrase = id2catname[cat]

            cur_img = {
                "file_name": filename,
                "height": imid2data[int(image_id)]["height"],
                "width": imid2data[int(image_id)]["width"],
                "id": next_img_id,
                "original_id": image_id,
                "coco_url": imid2data[int(image_id)]["coco_url"],
                "caption": phrase,
                "tokens_negative": [(0, len(phrase))],
                "dataset_name": d_name,
            }

            next_img_id += 1
            images.append(cur_img)

    print("Final number of datapoints:", len(images))
    ds = {"info": [], "licenses": [], "images": images, "annotations": annotations, "categories": categories}
    name = "lvis" if fract == 1 else f"lvis{round(fract * 100)}"
    with open(output_path / f"finetune_{name}_{split}.json", "w") as j_file:
        json.dump(ds, j_file)
    return next_img_id, next_id


def main(args):
    data_path = Path(args.data_path)
    output_path = Path(args.out_path)

    os.makedirs(str(output_path), exist_ok=True)

    for split in ["train", "val"]:
        convert(split, data_path, output_path, args.subset_fract)


if __name__ == "__main__":
    main(parse_args())
