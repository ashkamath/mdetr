# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
"""
This script is used to create the pretraining val dataset.
data_path :  path to original refexp annotations to be downloaded from https://github.com/lichengunc/refer
"""
import argparse
import json
import os
import pickle
from pathlib import Path
import sys
PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from utils.spans import consolidate_spans
from utils.text import get_root_and_nouns


def parse_args():
    parser = argparse.ArgumentParser("Conversion script")

    parser.add_argument(
        "--data_path",
        required=True,
        type=str,
        help="Path to the refexp data",
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


def convert(dataset_path: Path, split: str, output_path, coco_path, next_img_id: int = 0, next_id: int = 0):
    """Do the heavy lifting on the given split (eg 'train')"""

    print(f"Exporting {split}...")

    with open(f"{coco_path}/annotations/instances_train2014.json", "r") as f:
        coco_annotations = json.load(f)
    coco_images = coco_annotations["images"]
    coco_anns = coco_annotations["annotations"]
    annid2cocoann = {item["id"]: item for item in coco_anns}
    imgid2cocoimgs = {item["id"]: item for item in coco_images}

    categories = coco_annotations["categories"]
    annotations = []
    images = []

    for dataset_name in ["refcoco/refs(unc).p", "refcoco+/refs(unc).p", "refcocog/refs(umd).p"]:
        d_name = dataset_name.split("/")[0]

        with open(dataset_path / dataset_name, "rb") as f:
            data = pickle.load(f)

        for item in data:
            if item["split"] != split:
                continue

            for s in item["sentences"]:
                refexp = s["sent"]
                _, _, root_spans, neg_spans = get_root_and_nouns(refexp)
                root_spans = consolidate_spans(root_spans, refexp)
                neg_spans = consolidate_spans(neg_spans, refexp)

                filename = "_".join(item["file_name"].split("_")[:-1]) + ".jpg"
                cur_img = {
                    "file_name": filename,
                    "height": imgid2cocoimgs[item["image_id"]]["height"],
                    "width": imgid2cocoimgs[item["image_id"]]["width"],
                    "id": next_img_id,
                    "original_id": item["image_id"],
                    "caption": refexp,
                    "dataset_name": d_name,
                    "tokens_negative": neg_spans,
                }

                cur_obj = {
                    "area": annid2cocoann[item["ann_id"]]["area"],
                    "iscrowd": annid2cocoann[item["ann_id"]]["iscrowd"],
                    "image_id": next_img_id,
                    "category_id": item["category_id"],
                    "id": next_id,
                    "bbox": annid2cocoann[item["ann_id"]]["bbox"],
                    # "segmentation": annid2cocoann[item['ann_id']]['segmentation'],
                    "original_id": item["ann_id"],
                    "tokens_positive": root_spans,
                }
                next_id += 1
                annotations.append(cur_obj)
                next_img_id += 1
                images.append(cur_img)

    ds = {
        "info": coco_annotations["info"],
        "licenses": coco_annotations["licenses"],
        "images": images,
        "annotations": annotations,
        "categories": coco_annotations["categories"],
    }
    with open(output_path / f"final_refexp_val.json", "w") as j_file:
        json.dump(ds, j_file)
    return next_img_id, next_id


def main(args):
    data_path = Path(args.data_path)
    output_path = Path(args.out_path)

    os.makedirs(str(output_path), exist_ok=True)

    convert(data_path, "val", output_path, args.coco_path, 0, 0)


if __name__ == "__main__":
    main(parse_args())
