# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
"""
This script is a first step to create the combined pretraining dataset.
data_path :  path to original refexp annotations to be downloaded from https://github.com/lichengunc/refer
"""

import argparse
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import List
import sys
PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from tqdm import tqdm
from utils.dump import Annotation, Datapoint
from utils.spans import consolidate_spans
from utils.text import get_root_and_nouns


def parse_args():
    parser = argparse.ArgumentParser("Conversion script")

    parser.add_argument(
        "--refexp_ann_path", type=str, required=True,
    )
    parser.add_argument(
        "--out_path",
        default="",
        type=str,
        help="Path where to export the resulting dataset. ",
    )

    parser.add_argument(
        "--coco_path",
        required=True,
        type=str,
        help="Path to coco 2014 dataset.",
    )

    return parser.parse_args()


def convert(dataset_path: Path, split, output_path, coco_path):
    all_datapoints: List[Datapoint] = []

    safe_ids = set()
    with open(
        "combined_ref_exp_safe_train_ids.txt", "r"
    ) as f:
        for line in f:
            safe_ids.add(int(line.strip()))

    with open(f"{coco_path}/annotations/instances_train2014.json", "r") as f:
        coco_annotations = json.load(f)
    coco_anns = coco_annotations["annotations"]
    annid2cocoann = {item["id"]: item for item in coco_anns}

    for dataset_name in ["refcoco/refs(unc).p", "refcoco+/refs(unc).p", "refcocog/refs(umd).p"]:
        d_name = dataset_name.split("/")[0]

        with open(dataset_path / dataset_name, "rb") as f:
            data = pickle.load(f)

        for item in tqdm(data):
            if item["split"] != split:
                continue
            if item["split"] == "train" and item["image_id"] not in safe_ids:
                continue
            for s in item["sentences"]:
                refexp = s["sent"]
                target_bbox = annid2cocoann[item["ann_id"]]["bbox"]
                converted_bbox = [
                    target_bbox[0],
                    target_bbox[1],
                    target_bbox[2] + target_bbox[0],
                    target_bbox[3] + target_bbox[1],
                ]

                _, _, root_spans, neg_spans = get_root_and_nouns(refexp)
                cur_datapoint = Datapoint(
                    image_id=item["image_id"],
                    dataset_name=d_name,
                    original_id=item["ann_id"],
                    caption=refexp,
                    annotations=[],
                    tokens_negative=consolidate_spans(neg_spans, refexp),
                )

                cur_obj = Annotation(
                    area=annid2cocoann[item["ann_id"]]["area"],
                    iscrowd=annid2cocoann[item["ann_id"]]["iscrowd"],
                    category_id=item["category_id"],
                    bbox=target_bbox,
                    giou_friendly_bbox=converted_bbox,
                    tokens_positive=consolidate_spans(root_spans, refexp),
                )
                cur_datapoint.annotations.append(cur_obj)
                all_datapoints.append(cur_datapoint)

    with open(output_path / "refexp_dict.pkl", "wb") as f:
        pickle.dump(all_datapoints, f)


def main(args):
    dataset_path = Path(args.refexp_ann_path)
    output_path = Path(args.out_path) if args.out_path is not None else dataset_path

    os.makedirs(str(output_path), exist_ok=True)

    convert(dataset_path=dataset_path, split="train", output_path=output_path, coco_path=args.coco_path)


if __name__ == "__main__":
    main(parse_args())
