# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
import json
from typing import Any, List, NamedTuple, Optional, Tuple


class Annotation(NamedTuple):
    area: float
    iscrowd: int
    category_id: int
    bbox: List[float]
    giou_friendly_bbox: List[float]
    tokens_positive: List[Tuple[int, int]]


class Datapoint(NamedTuple):
    image_id: int
    dataset_name: str
    tokens_negative: List[Tuple[int, int]]
    original_id: int
    caption: str
    annotations: List[Annotation]


def convert2dataset_combined(
    datapoint_list_coco: List[Datapoint],
    datapoint_list_vg: List[Datapoint],
    imgid2imginfo_coco,
    imgid2imginfo_vg,
    output_path,
):
    """"""
    print(f"Dumping combined coco and vg images related all training examples...")
    next_img_id = 0
    next_id = 0

    annotations = []
    images = []

    for datapoint in datapoint_list_coco:
        img_id = datapoint.image_id
        filename = imgid2imginfo_coco[img_id]["file_name"]
        cur_img = {
            "file_name": filename,
            "height": imgid2imginfo_coco[img_id]["height"],
            "width": imgid2imginfo_coco[img_id]["width"],
            "id": next_img_id,
            "original_id": img_id,
            "caption": datapoint.caption,
            "tokens_negative": datapoint.tokens_negative,
            "data_source": "coco",
            "dataset_name": datapoint.dataset_name,
        }

        for anns in datapoint.annotations:
            cur_obj = {
                "area": float(anns.area),
                "iscrowd": anns.iscrowd,
                "image_id": next_img_id,
                "category_id": anns.category_id,
                "id": next_id,
                "bbox": anns.bbox,
                "tokens_positive": anns.tokens_positive,
            }
            next_id += 1
            annotations.append(cur_obj)

        next_img_id += 1
        images.append(cur_img)

    for datapoint in datapoint_list_vg:
        img_id = datapoint.image_id
        filename = f"{img_id}.jpg"
        cur_img = {
            "file_name": filename,
            "height": imgid2imginfo_vg[img_id]["height"],
            "width": imgid2imginfo_vg[img_id]["width"],
            "id": next_img_id,
            "original_id": img_id,
            "caption": datapoint.caption,
            "tokens_negative": datapoint.tokens_negative,
            "data_source": "vg",
            "dataset_name": datapoint.dataset_name,
        }

        for anns in datapoint.annotations:
            cur_obj = {
                "area": float(anns.area),
                "iscrowd": anns.iscrowd,
                "image_id": next_img_id,
                "category_id": anns.category_id,
                "id": next_id,
                "bbox": anns.bbox,
                "tokens_positive": anns.tokens_positive,
            }
            next_id += 1
            annotations.append(cur_obj)

        next_img_id += 1
        images.append(cur_img)

    ds = {"info": [], "licenses": [], "images": images, "annotations": annotations, "categories": []}
    with open(output_path / f"final_mixed_train.json", "w") as j_file:
        json.dump(ds, j_file)
    return next_img_id, next_id
