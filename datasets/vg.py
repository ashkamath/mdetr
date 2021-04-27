# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

from transformers import RobertaTokenizerFast

from .coco import ModulatedDetection, make_coco_transforms


class VGDetection(ModulatedDetection):
    pass


def build(image_set, args):
    img_dir = Path(args.vg_img_path)
    assert img_dir.exists(), f"provided VG img path {img_dir} does not exist"

    ann_file = Path(args.gqa_ann_path) / f"final_vg_{image_set}.json"

    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_encoder_type)
    dataset = VGDetection(
        img_dir,
        ann_file,
        transforms=make_coco_transforms(image_set, cautious=True),
        return_masks=args.masks,
        return_tokens=True,
        tokenizer=tokenizer,
    )
    return dataset
