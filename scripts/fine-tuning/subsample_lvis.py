# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
"""This scripts subsample LVIS in the same way as how we do it for MDETR.

This allows to train regular detectors (Mask-RCNN, DETR,...) for comparison on the few-shot setting.

It outputs the dataset in coco format.
"""
import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .lvis_coco_format import get_subset
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


def convert(split, data_path, output_path, fract):

    print("Loading annotations...")
    with open(data_path / f"lvis_v1_{split}.json", "r") as f:
        data = json.load(f)

    imid2data = {x["id"]: x for x in data["images"]}
    kept_ids = set(get_subset(data, fract if split == "train" else 1, imid2data))

    images = [im for im in data["images"] if im["id"] in kept_ids]
    annotations = [ann for ann in data["annotations"] if ann["image_id"] in kept_ids]

    print("Final number of datapoints:", len(images), len(annotations))
    ds = {
        "info": data["info"],
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": data["categories"],
    }
    name = "lvis" if fract == 1 else f"lvis{round(fract * 100)}"
    with open(output_path / f"detection_{name}_{split}.json", "w") as j_file:
        json.dump(ds, j_file)


def main(args):
    data_path = Path(args.data_path)
    output_path = Path(args.out_path)

    os.makedirs(str(output_path), exist_ok=True)

    for split in ["train", "val"]:
        convert(split, data_path, output_path, args.subset_fract)


if __name__ == "__main__":
    main(parse_args())
