# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
import argparse
import json
import os
import pickle
import re
from collections import defaultdict
from pathlib import Path

import multimodal
import numpy as np
from tqdm import tqdm

seed = 42


def parse_args():
    parser = argparse.ArgumentParser("Conversion script")

    parser.add_argument(
        "--data_path",
        default="",
        required=True,
        type=str,
        help="Path to the vqa dataset",
    )
    parser.add_argument(
        "--img_path",
        default="",
        required=True,
        type=str,
        help="Path to the vqa image dataset",
    )

    parser.add_argument(
        "--out_path",
        default="",
        type=str,
        help="Path where to export the resulting dataset.",
    )

    parser.add_argument(
        "--coco_path",
        default="",
        required=True,
        type=str,
        help="Path to coco dataset.",
    )
    return parser.parse_args()


def split_val(dataset):

    image_ids = set([x["image_id"] for x in dataset])
    minival_image_ids = np.random.choice(list(image_ids), 2000, replace=False)

    minival_dataset = []
    train_val_dataset = []
    for item in dataset:
        if item["image_id"] in minival_image_ids:
            minival_dataset.append(item)
        else:
            train_val_dataset.append(item)

    return train_val_dataset, minival_dataset


def convert(split, data_path, output_path, coco_path):

    dataset = list(multimodal.datasets.VQA2(dir_data=data_path, min_ans_occ=9, split=split))

    print(f"Dumping {split}...")
    next_img_id = 0
    next_id = 0

    if split in ["train"]:
        iminfo_files = ["instances_train2014"]
    elif split in ["val"]:
        iminfo_files = ["instances_val2014"]
    elif split in ["test-dev"]:
        iminfo_files = ["image_info_test-dev2015", "image_info_test2015"]
    elif split in ["test"]:
        iminfo_files = ["image_info_test2015"]
    else:
        assert False, f"Split {split} not recognized"

    imid2data = {}
    for iminfo_file in iminfo_files:
        with open(f"{coco_path}/annotations/{iminfo_file}.json", "r") as f:
            iminfo = json.load(f)
            imid2data.update({x["id"]: x for x in iminfo["images"]})

    if split == "val":
        trainval_dataset, minival_dataset = split_val(dataset)
        datasets = {"trainval": trainval_dataset, "minival": minival_dataset}
    else:
        datasets = {split: dataset}

    for dset_name, dset in datasets.items():

        categories = [{"supercategory": "object", "id": 1, "name": "object"}]
        annotations = []
        images = []
        d_name = "vqa2"

        for idx, datum in tqdm(enumerate(dset)):

            image_id = datum["image_id"]

            if dset_name in ["train", "trainval", "minival"]:

                cur_img = {
                    "file_name": imid2data[image_id]["file_name"],
                    "height": imid2data[image_id]["height"],
                    "width": imid2data[image_id]["width"],
                    "id": next_img_id,
                    "original_id": image_id,
                    "caption": datum["question"],
                    "question_id": datum["question_id"],
                    "answer": datum["multiple_choice_answer"],
                    "scores": datum["scores"],
                    "answer_type": datum["answer_type"],
                    "tokens_negative": [(0, len(datum["question"]))],
                    "dataset_name": d_name,
                }

            else:

                cur_img = {
                    "file_name": imid2data[image_id]["file_name"],
                    "height": imid2data[image_id]["height"],
                    "width": imid2data[image_id]["width"],
                    "id": next_img_id,
                    "original_id": image_id,
                    "caption": datum["question"],
                    "question_id": datum["question_id"],
                    "answer": None,
                    "scores": None,
                    "answer_type": None,
                    "tokens_negative": [(0, len(datum["question"]))],
                    "dataset_name": d_name,
                }

            next_img_id += 1
            images.append(cur_img)

        ds = {"info": [], "licenses": [], "images": images, "annotations": annotations, "categories": categories}

        print("Writing to file....")
        with open(output_path / f"finetune_vqa2_{dset_name}.json", "w") as j_file:
            json.dump(ds, j_file)
        print("Done!")

    return next_img_id, next_id


def main(args):
    data_path = Path(args.data_path)
    output_path = Path(args.out_path) if args.out_path is not None else data_path
    np.random.seed(seed)
    os.makedirs(str(output_path), exist_ok=True)

    for split in ["train", "val", "test-dev", "test"]:
        convert(split, data_path, output_path, args.coco_path)


if __name__ == "__main__":
    main(parse_args())
