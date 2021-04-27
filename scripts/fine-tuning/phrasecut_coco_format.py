# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
import argparse
import json
import os
import pickle
import re
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser("Conversion script")

    parser.add_argument(
        "--data_path",
        required=True,
        type=str,
        help="Path to the gqa dataset",
    )
    parser.add_argument(
        "--img_path",
        required=True,
        type=str,
        help="Path to the gqa image dataset",
    )

    parser.add_argument(
        "--out_path",
        default="",
        type=str,
        help="Path where to export the resulting dataset. Leave it to None to use the same path as above",
    )
    return parser.parse_args()


def convert(split, data_path, output_path, imid2data):

    with open(data_path / f"refer_{split}.json", "r") as f:
        data = json.load(f)

    img2ann = defaultdict(list)
    for datapoint in data:
        img2ann[datapoint["image_id"]].append(datapoint)

    print(f"Dumping {split}...")
    next_img_id = 0
    next_id = 0

    categories = [{"supercategory": "object", "id": 1, "name": "object"}]
    annotations = []
    images = []

    d_name = "phrasecut"

    for image_id, annotation_list in tqdm(img2ann.items()):

        for annotation in annotation_list:
            phrase = annotation["phrase"]
            task_id = annotation["task_id"]
            filename = f"{image_id}.jpg"

            cur_img = {
                "file_name": filename,
                "height": imid2data[int(image_id)]["height"],
                "width": imid2data[int(image_id)]["width"],
                "id": next_img_id,
                "original_id": image_id,
                "caption": phrase,
                "tokens_negative": [(0, len(phrase))],
                "dataset_name": d_name,
                "task_id": task_id,
            }

            assert len(annotation["Polygons"]) == len(annotation["instance_boxes"])

            instance_polygons_flattened = []
            for instance_polygons_list in annotation[
                "Polygons"
            ]:  # as many polygons as number of boxes ie len(annotation['Polygons']) == len(annotation['instance_boxes'])
                for polygon in instance_polygons_list:
                    polygon_flattened = []
                    for xy in polygon:
                        polygon_flattened.extend(xy)
                instance_polygons_flattened.append(polygon_flattened)

            assert len(instance_polygons_flattened) == len(
                annotation["instance_boxes"]
            ), "Number of combined polygons must be equal to the number of boxes"

            if len(annotation["instance_boxes"]) > 0:

                for i, target_bbox in enumerate(annotation["instance_boxes"]):
                    x, y, w, h = target_bbox
                    cur_obj = {
                        "area": h * w,
                        "iscrowd": 0,
                        "category_id": 1,
                        "bbox": target_bbox,
                        "segmentation": [instance_polygons_flattened[i]],
                        "tokens_positive": [(0, len(phrase))],
                        "image_id": next_img_id,
                        "id": next_id,
                    }

                    next_id += 1
                    annotations.append(cur_obj)

            next_img_id += 1
            images.append(cur_img)

    ds = {"info": [], "licenses": [], "images": images, "annotations": annotations, "categories": categories}
    with open(output_path / f"finetune_phrasecut_{split}.json", "w") as j_file:
        json.dump(ds, j_file)
    return next_img_id, next_id


def main(args):
    data_path = Path(args.data_path)
    output_path = Path(args.out_path)

    with open(data_path / "image_data_split.json", "r") as f:
        imdata = json.load(f)
    imid2data = {x["image_id"]: x for x in imdata}

    os.makedirs(str(output_path), exist_ok=True)

    # Phrasecut has 4 splits: train val miniv and test
    for split in ["miniv", "train", "val", "test"]:
        convert(split, data_path, output_path, imid2data)


if __name__ == "__main__":
    main(parse_args())
